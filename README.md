# py_template

This is a template for working with python and a virtual env. Relevant scripts are contained within the ```dev``` directory. 

## virtualenv

After initialization, the python environment will be located in ```python_env```. Run ```source python_env/bin/activate``` to activate and ```deactivate``` to deactivate.

## Scripts

### General scripts

```dev/create.sh```    Runs the creation wizard. Overriddes ``project.json`` and creates the source directory with the same name as the project name. Then runs ```dev/init.sh```

```dev/init.sh```   Initializes the project after creation or cloning. Creates the virtualenv, installs requirements, and initializes git hooks.

### After activation of virtualenv

```dev/freeze.sh``` Writes currently installed requirements into ```requirements.txt```. 

```dev/test.sh```   Runs pytest.

```dev/install_git_hooks.sh```  Should be called after a new script has been added to ```dev/hooks``` in order to register the script as a hook.

## Git hooks

Git hooks are located in ```dev/hooks```. 
After successful initialization using ```dev/init.sh```, a pre-commit hook should have been added. This hook should run tests, then freeze requirements, and should then add the new requirements.txt to the commit.  
