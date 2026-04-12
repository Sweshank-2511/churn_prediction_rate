from setuptools import find_packages, setup

def get_requirements(file_path):
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="CHURN_PREDICTION_RATE",
    version="0.0.1",
    author="Sweshank Kumar",
    author_email="sweshanksharma@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)