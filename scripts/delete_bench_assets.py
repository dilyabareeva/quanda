"""Delete all HuggingFace Hub repos under quanda-bench-test."""

from huggingface_hub import HfApi, list_datasets, list_models

NAMESPACE = "quanda-bench-test"


def main():
    api = HfApi()

    # List all model repos
    models = list(list_models(author=NAMESPACE))
    # List all dataset repos
    datasets = list(list_datasets(author=NAMESPACE))

    if not models and not datasets:
        print("No repos found under quanda-bench-test.")
        return

    print(f"Found {len(models)} model(s) and {len(datasets)} dataset(s):\n")
    for m in models:
        print(f"  [model]   {m.id}")
    for d in datasets:
        print(f"  [dataset] {d.id}")

    confirm = input("\nDelete all? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        return

    """
    for m in models:
        print(f"Deleting model {m.id} ...")
        api.delete_repo(repo_id=m.id, repo_type="model")
    """
    for d in datasets:
        if "metadata" not in d.id:
            continue
        print(f"Deleting dataset {d.id} ...")
        api.delete_repo(repo_id=d.id, repo_type="dataset")

    print("Done.")


if __name__ == "__main__":
    main()
