# Contributing to Coach

The following is a set of guidelines for contributing to Coach.
We'd like Coach to be useful to students, data scientists and researchers, and the purpose of these guidelines is to help maintain a level of quality and reliability for the Coach users community. 
Thanks for taking the time to contribute!

## Proposing a PR
If you would like to make code changes to Coach, whether adding new functionality or fixing a bug, please make sure to follow this list:
1. Add unit tests to any new component, or update existing tests for modified component’s functionality
2. Make sure [regression tests](https://github.com/NervanaSystems/coach/tree/master/rl_coach/tests) are passing. See the [Testing section](#Testing) for more details on the Coach testing methodology 
3. Update documentation to reflect any API changes, or added algorithm or environment. See the [Documentation section](#Documentation) for more details on Coach documentation. 
4. Adding an algorithm? 
    1. Please follow the guidelines [here](https://nervanasystems.github.io/coach/contributing/add_agent.html)
    2. Add a [benchmark](https://github.com/NervanaSystems/coach/blob/master/benchmarks/README.md) showing the results match those of the relevant research paper
    3. Update the algorithms diagram (https://github.com/NervanaSystems/coach/blob/master/docs_raw/source/diagrams.xml), export as png and update the README image (https://github.com/NervanaSystems/coach/blob/master/img/algorithms.png) 
5. Adding an environment? 
    1. Please follow the guidelines here (https://nervanasystems.github.io/coach/contributing/add_env.html)
    2. (Nice to have) Create a preset of an agent solving that environment


## Filing an Issue
Before filing an issue, please make sure you:
1. Go over the issues list to make sure a similar issue does not exist
2. Specify the following details:
    1. Setup - operating system, versions of installed environments or packages (if relevant), hardware details (if relevant)
    2. Use case description - command line, parameters
We prioritize issues to P1/P2/P3 based on our understanding of their impact on the Coach users community. It would be helpful if you describe the exact impact of the issue if not fixed


## Testing
Coach uses the pytest framework for its tests. All tests are located in the [tests folder](https://github.com/NervanaSystems/coach/tree/master/rl_coach/tests), and are divided to four categories - Unit, Integration, Golden and Trace tests.
Please refer to the tests [README](https://github.com/NervanaSystems/coach/blob/master/rl_coach/tests/README.md) for the full details of the Coach testing methodology.
Before proposing any code changes to Coach please make sure any new functionality is tested, and that regression tests are passing.


## Documentation
Most Coach’s documentation is located in the [Coach GitHub Pages](https://nervanasystems.github.io/coach/). The pages contain information on Coach’s usage and features, design and components, and on how to add new agents or environments. 
Their content is located in the [docs folder](https://github.com/NervanaSystems/coach/tree/master/docs), which is built from the raw files under the [docs_raw folder](https://github.com/NervanaSystems/coach/tree/master/docs_raw). 
When making documentation changes, please modify the files in the docs_raw folder, and refer to the [README](https://github.com/NervanaSystems/coach/blob/master/docs_raw/README.md) for instructions on how to build the docs folder. 
Both the docs_raw and docs folders need to be committed when documentation changes are made.  


## Contact
You’re welcome to contact the Intel Coach team either by filing a GitHub issue, or at coach@intel.com
