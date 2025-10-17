'''
merge enron_emails files to a single enron.txt
'''
import os
import logging

def collect_txt_files_recursively(root_dir: str):
    """Return a sorted list of absolute paths to all .txt files under root_dir.

    Sorting is done on the relative path to ensure deterministic output.
    """
    txt_files = []
    for current_root, _dirs, files in os.walk(root_dir):
        for filename in files:
            filepath = os.path.join(current_root, filename)
            if os.path.isdir(filepath):
                txt_files.extend(collect_txt_files_recursively(filepath))
            elif os.path.isfile(filepath):
                txt_files.append(os.path.join(current_root, filename))
    txt_files.sort(key=lambda p: os.path.relpath(p, root_dir))
    return txt_files


def merge_txt_files(input_root: str, output_file: str) -> int:
    """Merge all .txt files under input_root into output_file.

    Returns the number of files merged.
    """
    files_to_merge = collect_txt_files_recursively(input_root)
    if not files_to_merge:
        # Ensure we still create/overwrite the output file even if empty
        open(output_file, "w", encoding="utf-8").close()
        return 0

    with open(output_file, "w", encoding="utf-8") as writer:
        for idx, file_path in enumerate(files_to_merge):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as reader:
                content = reader.read()
            # Ensure each file ends with a single newline to separate documents
            if content and not content.endswith("\n"):
                content += "\n"
            writer.write(content)
            # Avoid adding extra blank lines beyond one separator
            if idx != len(files_to_merge) - 1:
                # Ensure at least one newline between files
                writer.write("")
    return len(files_to_merge)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    enron_root = os.path.join(base_dir, "enron_emails")
    output_path = os.path.join(base_dir, "enron.txt")

    os.makedirs(base_dir, exist_ok=True)
    merged_count = merge_txt_files(enron_root, output_path)
    # Be quiet by default; uncomment to print a small report
    # print(f"Merged {merged_count} files into {output_path}")
