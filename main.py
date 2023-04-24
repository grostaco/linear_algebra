import typer


def main(command: str):
    typer.echo(command)


if __name__ == '__main__':
    typer.run(main)
