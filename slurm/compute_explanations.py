from argparse import ArgumentParser


def compute_explanations(method):
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--method', required=True, type=int)
    args = parser.parse_args()
    main(args.model_id)