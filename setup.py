
from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    try:
        with open(file_path, encoding='utf-8') as f:
            # strip whitespace, ignore empty lines and comments
            requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    except FileNotFoundError:
        return []

    # remove editable reference to this package if present
    if '-e .' in requirements:
        requirements.remove('-e .')

    return requirements


if __name__ == '__main__':
    setup(
        name='ml_pipeline_project',
        version='0.1',
        description='ML Pipeline Project',
        author='Deeksha V',
        author_email='ed23b015@smail.iitm.ac.in',
        packages=find_packages(),
        install_requires=get_requirements('requirements.txt'),
        include_package_data=True,
    )