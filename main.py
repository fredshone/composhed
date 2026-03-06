"""CompSched entry point — delegates to train or generate subcommands."""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|generate] [args...]")
        print("  Or use: compsched-train / compsched-generate")
        sys.exit(1)

    cmd = sys.argv.pop(1)
    if cmd == "train":
        from composhed.train import main as train_main
        train_main()
    elif cmd == "generate":
        from composhed.generate import main as generate_main
        generate_main()
    elif cmd == "train-mdcev":
        from composhed.train_mdcev import main as train_mdcev_main
        train_mdcev_main()
    elif cmd == "generate-mdcev":
        from composhed.generate_mdcev import main as generate_mdcev_main
        generate_mdcev_main()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
