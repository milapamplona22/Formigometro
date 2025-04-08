# LabCog #
## Repository with codes and examples for performing statistical analyses ##

1. Rules
2. Requirements
3. Basic Usage


### 1. Rules ###
1. Keep the **master branch always functional**
2. When **modifying** something existing or **adding** something new, do so in a **separate branch**. Once you ensure that this new branch is working and in its final version, only then merge it with the master branch to make it available to everyone, and delete the branch that was created for development.
3. Files with **examples** of analyses should be named with 'example_' at the beginning.
4. Files with **data** should be named with 'data_' at the beginning.
5. Provide **references** to websites, books, and papers, both about the explanation of the analysis and the code.
6. If creating an example in Jupyter **notebook**, also add the exported .html file.


### 2. What is required? (Requirements) ###

* Install [git](https://git-scm.com)
* Depending on the code you are going to use: Python, R, Matlab (/Octave)
* It is recommended to have [Jupyter Notebook](http://jupyter.org) to run interactive examples (.ipynb) (Jupyter requires Python)

### 3. Basic Usage (= creating a local copy)###
Simple reference for using git: http://rogerdudler.github.io/git-guide/

1. Download the current version of the repository:
* If this is your first time, you still need to clone (download) the repository:
```
#!git
     git clone https:\address up there on the top right of this page (where it says HTTPS)
```
* If you already have the repository, go to it and download the latest version
```
#!git
     git pull origin master
```
 All changes you make will only be applied to your machine.


2. If you want to discard your changes and download it again:
```
#!git
     git fetch --all
     git reset --hard origin/master
```
