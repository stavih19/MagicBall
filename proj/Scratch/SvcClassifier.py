import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import joblib
from sklearn.svm import SVC
from torchvision.models import ResNet18_Weights
import random

# Load a pre-trained ResNet model and remove the last layer
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()  # Set the model to evaluation mode

# Transform the image to fit the ResNet input requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # Define combined transform
# combined_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#
#     # Lambda function to apply grayscale transform conditionally
#     transforms.Lambda(lambda img: img.convert("L").convert("RGB") if random.random() < 0.5 else img),
#
#     # ColorJitter for color augmentation (also applies to grayscale since it's converted back to 3 channels)
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
# ])


def extract_features(img):
    tr = transform(img)
    img_t = tr.unsqueeze(0)

    with torch.no_grad():
        features = model(img_t).squeeze().numpy()  # Extract features

    return features  # This will be a (512,) feature vector


def train_model():
    # Path to the folder containing images
    image_paths = []
    labels = []

    folder_image_train_object = 'D:\\PythonProjects\\MagicBall\\Find the ball\\dataset\\color\\train\\object_color'
    image_files = [f for f in os.listdir(folder_image_train_object) if f.endswith('.JPG')]
    for image_name in image_files[0:]:
        image_paths.append(os.path.join(folder_image_train_object, image_name))
        labels.append(1)

    folder_image_train_no_object = 'D:\\PythonProjects\\MagicBall\\Find the ball\\dataset\\color\\train\\no_object_color'
    image_files = [f for f in os.listdir(folder_image_train_no_object) if f.endswith('.JPG')]
    for image_name in image_files[0:]:
        image_paths.append(os.path.join(folder_image_train_no_object, image_name))
        labels.append(0)

    folder_image_val_object = 'D:\\PythonProjects\\MagicBall\\Find the ball\\dataset\\color\\val\\object_color'
    image_files = [f for f in os.listdir(folder_image_val_object) if f.endswith('.JPG')]
    for image_name in image_files[0:]:
        image_paths.append(os.path.join(folder_image_val_object, image_name))
        labels.append(1)

    folder_image_val_no_object = 'D:\\PythonProjects\\MagicBall\\Find the ball\\dataset\\color\\val\\no_object_color'
    image_files = [f for f in os.listdir(folder_image_val_no_object) if f.endswith('.JPG')]
    for image_name in image_files[0:]:
        image_paths.append(os.path.join(folder_image_val_no_object, image_name))
        labels.append(0)

    # Extract features from each image
    features = [extract_features(Image.open(img_path)) for img_path in image_paths]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    svc_classifier = SVC(probability=True)
    svc_classifier.fit(X_train, y_train)

    # Evaluate the model
    accuracy = svc_classifier.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(svc_classifier, 'svc_model.pkl')


def svc_pil_interference(img, svc_classifier):
    feature = extract_features(img)
    return svc_classifier.predict_proba(feature.reshape(1, -1))[:, 1][0]


def svc_interference(img, svc_classifier):
    pil_img = Image.fromarray(img.astype('uint8'), 'RGB')
    return svc_pil_interference(pil_img, svc_classifier)


if __name__ == '__main__':
    train_model()


    # image_path = 'D:\\PythonProjects\\MagicBall\\Find the ball\\Special Ball\\challenge_image.png'
    # img = Image.open(image_path)
    # svc_classifier = joblib.load('svc_model.pkl')
    # prediction = svc_pil_interference(img, svc_classifier)
    # print(prediction)
