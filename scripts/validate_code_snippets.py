import re
import sys
import textwrap


def extract_snippet_from_readme(text: str, snippet_id: str) -> str:
    """
    Extract code snippet from the README.

    Format:
        <!-- START<snippet_id> -->
        ... code ...
        <!-- END<snippet_id> -->
    """
    # Create a regex pattern
    pattern = re.compile(
        rf'<!--\s*START{re.escape(snippet_id)}\s*-->(.*?)<!--\s*END{re.escape(snippet_id)}\s*-->',
        re.DOTALL
    )

    # Find the snippet
    match = pattern.search(text)
    if not match:
        print(f"Could not find snippet {snippet_id} in README.")
        sys.exit(1)
    snippet = match.group(1)

    # Remove code fences:
    snippet = re.sub(r'```python', '', snippet)
    snippet = re.sub(r'```', '', snippet)

    # Remove indentation
    snippet = textwrap.dedent(snippet)

    return snippet.strip()


def extract_snippet_from_test(text: str, snippet_id: str) -> str:
    """
    Extract code snippet from the test file.

    Format:
        # START<snippet_id>
        ... code ...
        # END<snippet_id>
    """
    # Create a regex pattern
    pattern = re.compile(
        rf'#\s*START{re.escape(snippet_id)}\s*\n(.*?)\n\s*#\s*END{re.escape(snippet_id)}',
        re.DOTALL
    )

    # Find the snippet
    match = pattern.search(text)
    if not match:
        print(
            f"Could not find snippet {snippet_id} in test file.")
        sys.exit(1)
    snippet = match.group(1)

    # Remove indentation
    snippet = textwrap.dedent(snippet)

    return snippet.strip()


def main():
    snippet_ids = [
        "1", "2", "3", "4", "5", "6", "7_1", "7_2",
        "8", "9", "10", "11", "12", "13_2", "14"
    ]

    # Load the README file
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme_text = f.read()
    except FileNotFoundError:
        print("README.md not found!")
        sys.exit(1)

    # Load the test file
    try:
        with open("tests/integration/test_quickstart.py", "r", encoding="utf-8") as f:
            test_text = f.read()
    except FileNotFoundError:
        print("tests/integration/test_quickstart.py not found!")
        sys.exit(1)

    all_equal = True

    for snippet_id in snippet_ids:
        readme_snippet = extract_snippet_from_readme(readme_text, snippet_id)
        test_snippet = extract_snippet_from_test(test_text, snippet_id)

        if readme_snippet != test_snippet:
            print(f"Mismatch found in snippet {snippet_id}:")
            print("----- README snippet -----")
            print(readme_snippet)
            print("----- Test snippet -----")
            print(test_snippet)
            all_equal = False
        else:
            print(f"Snippet '{snippet_id}' matches.")

    if not all_equal:
        sys.exit(1)
    else:
        print("All code snippets match!")
        sys.exit(0)


if __name__ == "__main__":
    main()
