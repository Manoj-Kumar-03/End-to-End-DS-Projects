from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    '''
        This Function will return List of requirements
    '''
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n',' ') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
name="mlproject",
version="0.0.0.0",
author="Manoj",
author_email="lingala.manoj3@gmail.com",
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)