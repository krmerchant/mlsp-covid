import click
import classifier_fe
import classifier_prediction




@click.group()
def cli():
    pass


@cli.group()
def extract_features():
    pass



extract_features.add_command(classifier_fe.train)
extract_features.add_command(classifier_fe.test)
extract_features.add_command(classifier_fe.generate_features)


@cli.group()
def predict():
    pass


predict.add_command(classifier_prediction.train_svm)

if __name__ == '__main__':
    cli()
