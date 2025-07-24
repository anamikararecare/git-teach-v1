import os
from tree_sitter import Language, Parser

LANGUAGES = {'javascript': './tree-sitter-javascript'}

# building shared parser library
SO_PATH = 'build/my-languages.so'
if not os.path.exists(SO_PATH):
    Language.build_library(
        SO_PATH,
        list(LANGUAGES.values())
    )

# load language
JAVASCRIPT = Language(SO_PATH, 'javascript')
parser = Parser()
parser.set_language(JAVASCRIPT)

# chunking logic
def get_chunks(file_path: str, code: str, language: str='javascript'):
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    chunks = []

    def walk(node):
        noteworthy_types = {
            "function_declaration",
            "method_definition",
            "class_declaration",
            "lexical_declaration",
            "export_statement"
        }

        if node.type in noteworthy_types:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            chunks.append({
                'type': node.type,
                'name': get_node_name(node, code),
                'code': code.splitlines()[start_line - 1: end_line],
                'file': file_path
            })
        
        for child in node.children:
            walk(child)

    walk(root_node)
    return chunks

def get_node_name(node, code: str) -> str:
    for child in node.children:
        if child.type == "identifier":
            return code[child.start_byte:child.end_byte]
    return "anonymous"

def chunk_repo(repo_path: str) -> list[dict]:
    chunks = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if not file.endswith('.js'):
                continue
            file_path = os.path.join(root, file)
            try:
                chunks.extend(get_chunks(file_path))
            except Exception as e:
                print(f"Failed to parse {file_path}: {str(e)}")
    return chunks