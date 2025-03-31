import click
import classifier_fe
import classifier_prediction
import generate_dataset_csv
import utility

@click.group()
def cli():
    pass


@cli.group()
def dataset():
    pass

dataset.add_command(generate_dataset_csv.partition_dataset)


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

@cli.group()
def utils():
    pass

utils.add_command(utility.cuda_mem)



if __name__ == '__main__':
    cli()
