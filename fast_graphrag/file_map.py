import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser
import os
from collections import namedtuple
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import argparse

TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())


@dataclass(frozen=True)
class Scope:
    kind: str
    name: str
    start_pos: int
    end_pos: int
    identifiers: List[tuple[str, int]]  # (name, position) pairs
    children: List["Scope"]
    parent: Optional["Scope"] = None


Tag = namedtuple("Tag", "rel_fname fname line name kind".split())

FILEMAP_DESC = """
<filemap>
"""


class FileMapper:
    def __init__(self):
        self.ts_parser = Parser(TS_LANGUAGE)
        self.tsx_parser = Parser(TSX_LANGUAGE)
        self.seen = set()

        # self.rust_parser = Parser()

        # RUST_LANGUAGE = Language(language_dir, "rust")
        # self.rust_parser.set_language(RUST_LANGUAGE)

    def _get_identifier_name(self, node: Any) -> Optional[str]:
        """Extract identifier name from a node."""
        # Direct identifier types
        if node.type in {
            "identifier",
            "property_identifier",
            "private_property_identifier",
            "variable_declarator",
        }:
            return node.text.decode("utf-8")

        # For property assignments that are functions
        if node.type == "property_identifier":
            # Check if this is a property being assigned to a function
            next_sibling = node.next_sibling
            if next_sibling and next_sibling.type == "=":
                next_next = next_sibling.next_sibling
                if next_next and next_next.type in {"arrow_function", "function"}:
                    return node.text.decode("utf-8")

        # For declarations, look for specific name patterns
        for child in node.children:
            if child.type in {
                "identifier",
                "property_identifier",
                "private_property_identifier",
                "type_identifier",
            }:
                return child.text.decode("utf-8")

        return None

    def _collect_identifiers(
        self, node: Any, source_bytes: bytes, filepath: str
    ) -> tuple[list[tuple[str, int]], list[Scope]]:
        """Collect unique identifiers and their positions in current scope."""
        identifiers = set()
        scopes = []

        def visit(node):
            if node.type == "type_identifier":
                identifiers.add(node.text.decode("utf-8"))
                return

            if node.type == "generic_type":
                for child in node.children:
                    if child.type in {"type_identifier", "identifier"}:
                        identifiers.add(child.text.decode("utf-8"))
                return

            # Handle interface property declarations
            if node.type == "property_signature":
                # Get the property name
                for child in node.children:
                    if child.type in {"identifier", "property_identifier"}:
                        identifiers.add(child.text.decode("utf-8"))
                    # Get the type reference if it exists
                    elif child.type == "type_annotation":
                        for type_child in child.children:
                            if type_child.type == "type_identifier":
                                identifiers.add(type_child.text.decode("utf-8"))

            # Handle this.property references
            elif node.type == "member_expression":
                if node.children[0].type == "this":
                    prop = node.children[2]  # Get the property name after 'this.'
                    if prop.type in {"identifier", "property_identifier"}:
                        identifiers.add(prop.text.decode("utf-8"))
                    return

            elif node.type == "identifier":
                name = node.text.decode("utf-8")
                identifiers.add(name)
                return

            # Check if this node creates a new scope
            if scope := self._build_scope_tree(node, source_bytes, filepath):
                scopes.append(scope)

            # Continue traversing
            for child in node.children:
                visit(child)

        visit(node)
        return sorted(identifiers), scopes

    def _build_scope_tree(self, node: Any, source_bytes: bytes, filepath: str) -> Optional[Scope]:
        """Build scope tree focusing only on named declarations."""
        if node in self.seen:
            return None
        self.seen.add(node)

        def filter_scope_identifiers(identifiers, child_scopes):
            """Filter out identifiers that belong to child scopes."""
            child_identifiers = set()
            for child in child_scopes:
                # Add the child's name
                if child.name:
                    child_identifiers.add(child.name)
                # Add all identifiers from the child scope
                child_identifiers.update(name for name in child.identifiers)
                # Recursively add identifiers from nested scopes
                for nested_child in child.children:
                    child_identifiers.update(self._get_all_child_identifiers(nested_child))

            # Filter out identifiers that appear in child scopes
            return [name for name in identifiers if name not in child_identifiers]

        # List of node types that create named scopes
        scope_types = {
            "program": "file",
            "source_file": "file",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "function_declaration": "function",
            "method_definition": "method",
            "namespace_declaration": "namespace",
            "module_declaration": "module",
            "type_alias_declaration": "type",
        }

        if node.type == "property_identifier" or node.type == "private_property_identifier":
            next_sibling = node.next_sibling
            if next_sibling and next_sibling.type == "=":
                next_next = next_sibling.next_sibling
                if next_next and next_next.type == "arrow_function":
                    name = node.text.decode("utf-8")
                    identifiers, child_scopes = self._collect_identifiers(next_next, source_bytes, filepath)

                    filtered_identifiers = filter_scope_identifiers(identifiers, child_scopes)

                    return Scope(
                        kind="method",
                        name=name,
                        start_pos=node.start_byte,
                        end_pos=next_next.end_byte,
                        identifiers=filtered_identifiers,
                        children=child_scopes,
                    )

        # Handle variable declarations assigned to arrow functions
        if node.type == "variable_declarator":
            # Get the name from the declarator
            name = None
            arrow_function = None
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8")
                elif child.type == "arrow_function":
                    arrow_function = child

            if name and arrow_function:
                identifiers, child_scopes = self._collect_identifiers(arrow_function, source_bytes, filepath)

                filtered_identifiers = filter_scope_identifiers(identifiers, child_scopes)

                return Scope(
                    kind="function",
                    name=name,
                    start_pos=node.start_byte,
                    end_pos=arrow_function.end_byte,
                    identifiers=filtered_identifiers,
                    children=child_scopes,
                )

        # Only create scopes for named declarations
        scope_kind = scope_types.get(node.type)
        if not scope_kind:
            return None

        # Get name if this is a named declaration
        name = self._get_identifier_name(node)
        if not name and node.type != "program" and node.type != "source_file":
            return None

        # For file scope, use filepath as name
        if scope_kind == "file":
            name = filepath

        # Collect identifiers and child scopes
        identifiers, child_scopes = self._collect_identifiers(node, source_bytes, filepath)

        filtered_identifiers = filter_scope_identifiers(identifiers, child_scopes)

        return Scope(
            kind=scope_kind,
            name=name,
            start_pos=node.start_byte,
            end_pos=node.end_byte,
            identifiers=filtered_identifiers,
            children=child_scopes,
        )

    def _get_all_child_identifiers(self, scope: Scope) -> set[str]:
        """Recursively get all identifiers from a scope and its children."""
        identifiers = set()
        if scope.name:
            identifiers.add(scope.name)
        identifiers.update(name for name in scope.identifiers)
        for child in scope.children:
            identifiers.update(self._get_all_child_identifiers(child))
        return identifiers

    def generate_map(self, file_path: str, relative_file_path: str) -> Optional[Scope]:
        """Generate a scope map for a given file."""
        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            source_bytes = f.read()

        ext = Path(file_path).suffix
        parser = self.ts_parser if ext == ".ts" else self.tsx_parser

        tree = parser.parse(source_bytes)
        return self._build_scope_tree(tree.root_node, source_bytes, relative_file_path)

    def format_scope_chunks(self, scope: Scope, char_limit: int = 3600) -> list[str]:
        """Format scope tree as a list of chunks, each under char_limit."""
        chunks = []
        current_chunk = []
        current_length = 0
        open_scopes = []  # Stack of (scope, indent_level) tuples

        def add_to_chunk(line: str, indent: int):
            nonlocal current_length, current_chunk
            current_length += len(line) + 1  # +1 for newline
            current_chunk.append(("  " * indent) + line)

        def flush_chunk():
            nonlocal current_chunk, current_length
            if current_chunk:
                # Close all open scopes
                for _ in range(len(open_scopes)):
                    add_to_chunk("}", open_scopes[-1][1])

                chunk_text = FILEMAP_DESC + "\n" + "\n".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with reopened scopes
                current_chunk = []
                current_length = len(FILEMAP_DESC) + 1

                # Reopen all scopes that were open
                for scope, indent in open_scopes:
                    if scope.kind == "file":
                        line = f"Filepath({scope.name}) {{"
                    else:
                        line = f"{scope.kind} {scope.name} {{"
                    add_to_chunk(line, indent)

        def format_scope_recursive(scope: Scope, indent: int):
            nonlocal current_length, current_chunk, open_scopes

            # Check if adding this scope would exceed limit
            estimated_scope_size = 100  # Base size for scope declaration
            if scope.identifiers:
                estimated_scope_size += len(", ".join(scope.identifiers)) + 100

            # If adding this would exceed limit, flush chunk
            if current_length + estimated_scope_size > char_limit:
                flush_chunk()

            # Add scope opening
            if scope.kind == "file":
                add_to_chunk(f"Filepath({scope.name}) {{", indent)
            else:
                add_to_chunk(f"{scope.kind} {scope.name} {{", indent)

            open_scopes.append((scope, indent))

            # Add identifiers
            if scope.identifiers:
                add_to_chunk("identifiers = [", indent + 1)

                # Split identifiers if needed
                identifiers = scope.identifiers
                while identifiers:
                    # Calculate how many identifiers we can fit
                    current_line = ", ".join(identifiers)
                    if current_length + len(current_line) + 100 > char_limit:
                        # Find break point
                        for i in range(len(identifiers)):
                            partial_line = ", ".join(identifiers[:i])
                            if current_length + len(partial_line) + 100 > char_limit:
                                if i > 0:
                                    add_to_chunk(partial_line, indent + 2)
                                    identifiers = identifiers[i:]
                                    flush_chunk()
                                break
                    else:
                        add_to_chunk(current_line, indent + 2)
                        identifiers = []

                add_to_chunk("]", indent + 1)

            # Process children
            for child in scope.children:
                format_scope_recursive(child, indent + 1)

            # Close scope
            open_scopes.pop()
            add_to_chunk("}", indent)

        # Start formatting
        format_scope_recursive(scope, 0)

        # Flush final chunk
        if current_chunk:
            chunks.append(FILEMAP_DESC + "\n" + "\n".join(current_chunk))

        return chunks

    def format_scope(self, scope: Scope, char_limit: int = 3600) -> str:
        """Format scope tree as string, breaking into chunks if needed."""
        chunks = self.format_scope_chunks(scope, char_limit)
        return "\n\n=== CHUNK BREAK ===\n\n".join(chunks)


def main():
    mapper = FileMapper()
    parser = argparse.ArgumentParser(description="Extract condensed filemap from TS/TSX file")
    parser.add_argument("--path", required=True, type=str, help="File to analyse")
    args = parser.parse_args()

    scope_tree = mapper.generate_map(args.path, args.path)
    if scope_tree:
        print(mapper.format_scope(scope_tree))


if __name__ == "__main__":
    main()
