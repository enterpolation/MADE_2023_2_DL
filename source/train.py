import time
import warnings
import torch
import numpy as np
import config
import click

from metric import character_error_rate, decode
from model import CRNN
from dataset import ImageDataset

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")


def set_seed(seed=42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_inference(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn,
        opt_fn: torch.optim,
        metric=character_error_rate,
        device=config.DEVICE,
        training=True,
) -> (float, float):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_metric = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if training:
            opt_fn.zero_grad()

        predictions = model(images)

        loss = loss_fn(predictions, labels)

        running_loss += loss.item()

        if training:
            loss.backward()
            opt_fn.step()

        running_metric += metric(predictions, labels)

    epoch_loss = running_loss / len(dataloader)
    epoch_metric = running_metric / len(dataloader)

    return epoch_loss, epoch_metric


def train(
        save_path: str,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn,
        opt_fn: torch.optim,
        metric=character_error_rate,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE,
) -> None:
    print(f"Training on {device}...")

    for epoch in tqdm(range(num_epochs)):
        start = time.time()

        epoch_train_loss, epoch_train_metric = run_inference(
            dataloader=train_loader,
            model=model,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            metric=metric,
            device=device,
            training=True,
        )

        epoch_test_loss, epoch_test_metric = run_inference(
            dataloader=test_loader,
            model=model,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            metric=metric,
            device=device,
            training=False,
        )

        print(
            f"\nEpoch №{epoch + 1} | "
            f"Train loss: {epoch_train_loss:.4f} | "
            f"Test loss: {epoch_test_loss:.4f} | "
            f"Train CER: {epoch_train_metric:.4f} | "
            f"Test CER: {epoch_test_metric:.4f} | "
            f"Time: {time.time() - start:.4f} seconds"
        )

        torch.save(model.state_dict(), save_path)


def train_test_split(
        dataset: torch.utils.data.Dataset, test_size=0.2
) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    dataset_size = len(dataset)

    indices = torch.randint(dataset_size - 1, (dataset_size,))

    split = int(np.floor(test_size * dataset_size))

    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, sampler=test_sampler
    )

    return train_dataloader, test_dataloader


def print_errors(
        num_errors: int,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        criterion=character_error_rate,
        device=config.DEVICE,
) -> None:
    errors = []

    for i, (image, label) in enumerate(dataset):
        image = image.to(device)
        image = torch.unsqueeze(image, 0)
        label = torch.unsqueeze(label, 0)

        prediction = model(image).detach().cpu()
        metric = criterion(prediction, label).item()
        errors.append((metric, i, prediction[0]))

    print(f"Top-{num_errors} errors:")
    for error, i, prediction in sorted(errors, reverse=True)[:num_errors]:
        image, label = dataset[i]
        label = torch.unsqueeze(label, 0)
        prediction = torch.unsqueeze(prediction, 0)
        print(
            f"Ground truth: {decode(label)[0]}; prediction: {decode(prediction)[0]}; CER={error:.4f}"
        )


@click.command()
@click.option("-dp", "--data_path", help="Path to data.")
@click.option(
    "-mp", "--model_path", help="Path to store the serialized model (.pth)."
)
def main(data_path: str, model_path: str) -> None:
    # set random seed
    set_seed()

    # data transformations
    transform = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    # dataset
    dataset = ImageDataset(data_path, transform=transform)
    train_dataloader, test_dataloader = train_test_split(
        dataset, test_size=config.TEST_SIZE
    )

    # model setup
    net = CRNN()
    net = net.to(config.DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # train & evaluate
    train(
        save_path=model_path,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        model=net,
        loss_fn=criterion,
        opt_fn=optimizer,
        metric=character_error_rate,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE,
    )

    _, final_metric = run_inference(
        dataloader=test_dataloader,
        model=net,
        loss_fn=criterion,
        opt_fn=optimizer,
        metric=character_error_rate,
        device=config.DEVICE,
        training=False,
    )

    print(f"Finished training. Final metric on test set: {final_metric:.4f}.")

    print_errors(
        num_errors=8,
        dataset=dataset,
        model=net,
    )


if __name__ == "__main__":
    main()
