# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .create_index import create_index

def main():
    import typer
    import sys
    from wasabi import msg
    from spacy_ann.cli import create_index

    commands = {
        "create_index": create_index,
    }
    if len(sys.argv) == 1:
        msg.info("Available commands", ", ".join(commands), exits=1)
    command = sys.argv.pop(1)
    sys.argv[0] = "spacy_ann %s" % command
    if command in commands:
        typer.run(commands[command])
    else:
        available = "Available: {}".format(", ".join(commands))
        msg.fail("Unknown command: {}".format(command), available, exits=1)

if __name__ == "__main__":
    main()