import click
import training
import testing 





@click.group()
def cli():
    pass


cli.add_command(training.train)
cli.add_command(testing.test)

if __name__ == '__main__':
    cli()
