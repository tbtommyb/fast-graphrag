import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser
import os
from collections import namedtuple
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

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


class RepoMapper:
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
        return list(identifiers), scopes

    def _build_scope_tree(
        self, node: Any, source_bytes: bytes, filepath: str
    ) -> Optional[Scope]:
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
                    child_identifiers.update(
                        self._get_all_child_identifiers(nested_child)
                    )

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
        }

        if (
            node.type == "property_identifier"
            or node.type == "private_property_identifier"
        ):
            next_sibling = node.next_sibling
            if next_sibling and next_sibling.type == "=":
                next_next = next_sibling.next_sibling
                if next_next and next_next.type == "arrow_function":
                    name = node.text.decode("utf-8")
                    identifiers, child_scopes = self._collect_identifiers(
                        next_next, source_bytes, filepath
                    )

                    filtered_identifiers = filter_scope_identifiers(
                        identifiers, child_scopes
                    )

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
                identifiers, child_scopes = self._collect_identifiers(
                    arrow_function, source_bytes, filepath
                )

                filtered_identifiers = filter_scope_identifiers(
                    identifiers, child_scopes
                )

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
        identifiers, child_scopes = self._collect_identifiers(
            node, source_bytes, filepath
        )

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

    def generate_map(self, file_path: str) -> Optional[Scope]:
        """Generate a scope map for a given file."""
        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            source_bytes = f.read()

        ext = Path(file_path).suffix
        parser = self.ts_parser if ext == ".ts" else self.tsx_parser
        # language = TS_LANGUAGE if ext == ".ts" else TSX_LANGUAGE

        # query_scm = Path("grammars/typescript-tags.scm").read_text()
        tree = parser.parse(source_bytes)
        # query = language.query(query_scm)
        # captures = query.captures(tree.root_node)
        # saw = set()
        # all_nodes = list(captures)

        # for node, tag in all_nodes:
        #     if tag.startswith("name.definition."):
        #         kind = "def"
        #     elif tag.startswith("name.reference."):
        #         kind = "ref"
        #     else:
        #         continue

        #     saw.add(kind)

        #     result = Tag(
        #         rel_fname=file_path,
        #         fname=file_path,
        #         name=node.text.decode("utf-8"),
        #         kind=kind,
        #         line=node.start_point[0],
        #     )

        #     yield result

        # if "ref" in saw:
        #     return
        # if "def" not in saw:
        #     return

        # # We saw defs, without any refs
        # # Some tags files only provide defs (cpp, for example)
        # # Use pygments to backfill refs

        # try:
        #     lexer = guess_lexer_for_filename(fname, code)
        # except Exception:  # On Windows, bad ref to time.clock which is deprecated?
        #     # self.io.tool_error(f"Error lexing {fname}")
        #     return

        # tokens = list(lexer.get_tokens(code))
        # tokens = [token[1] for token in tokens if token[0] in Token.Name]

        # for token in tokens:
        #     yield Tag(
        #         rel_fname=rel_fname,
        #         fname=fname,
        #         name=token,
        #         kind="ref",
        #         line=-1,
        #     )
        return self._build_scope_tree(tree.root_node, source_bytes, file_path)

    def format_scope(self, scope: Scope, indent: int = 0) -> str:
        """Format scope tree as string."""
        result = []
        indent_str = "  " * indent

        if scope.kind == "file":
            result.append(f"{indent_str}File({scope.name}) {{")
        else:
            result.append(f"{indent_str}{scope.kind} {scope.name} {{")

        if scope.identifiers:
            result.append(f"{indent_str}  identifiers = [")
            result.append(f'{indent_str}    {", ".join(scope.identifiers)}')
            result.append(f"{indent_str}  ]")

        for child in scope.children:
            result.append(self.format_scope(child, indent + 1))

        result.append(f"{indent_str}}}")
        return "\n".join(result)


def main():
    # Initialize mapper
    mapper = RepoMapper()

    # Example usage
    file_path = "/Users/tomjhnsn/workplace/avlrc-dev/src/AVLivingRoomClient/packages/details/ui/components/DetailsPage.tsx"
    scope_tree = mapper.generate_map(file_path)
    if scope_tree:
        print(mapper.format_scope(scope_tree))


if __name__ == "__main__":
    main()
