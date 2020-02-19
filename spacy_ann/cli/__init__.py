# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

def main():
    import typer
    import sys
    from wasabi import msg
    from spacy_ann.cli.create_index import create_index
    from spacy_ann.cli.example_data import example_data
    from spacy_ann.cli.serve import serve

    commands = {
        "create_index": create_index,
        "example_data": example_data,
        "serve": serve
    }
    if len(sys.argv) == 1:
        msg.info("Available commands", ", ".join(commands), exits=1)
    command = sys.argv.pop(1)
    sys.argv[0] = f"spacy_ann {command}"
    if command in commands:
        typer.run(commands[command])
    else:
        available = "Available: {}".format(", ".join(commands))
        msg.fail("Unknown command: {}".format(command), available, exits=1)

if __name__ == "__main__":
    main()