# Coach - Tests

Coach is a complex framework consisting of various features and running schemes.
On top of that, reinforcement learning adds stochasticity in many places along the experiments, which makes getting the
same results run-after-run is almost impossible.
To address those issues, and ensure that Coach keeps working as expected, we separated our testing mechanism into
several parts, each testing the framework in different areas and strictness.

* **Docker** -
    
    The docker image we supply checks Coach in terms of installation process, and verifies that all the components
    are installed correctly. To build the Docke, use the command:
    
    ```
    docker build . -t coach
    docker run -it coach /bin/bash
    ```
    

* **Unit tests** -
    
    The unit tests test sub components of Coach with different parameters and verifies that they work as expected.
    There are currently tens of tests and we keep adding new ones. We use pytest in order to run the tests, using
    the following command:
    
    ```
    python3 -m pytest rl_coach/tests -m unit_test
    ```

* **Integration tests** -
    
    The integration tests make sure that all the presets are runnable. It's a static tests that does not check the
    performance at all. It only checks that the preset can start running with no import error or other bugs.
    To run the integration tests, use the following command:
    
    ```
    python3 -m pytest rl_coach/tests -m integration_test
    ```

* **Golden tests** -
    
    The golden tests run a subset of the presets available in Coach, and verify that they pass a known score after
    a known amount of steps. The threshold for the tests are defined as part of each preset. The presets which are
    tested are presets that can be run in a short amount of time, and the requirements for passing are quite weak.
    The golden tests can be run using the following command:
    
    ```
    python3 rl_coach/tests/golden_tests.py
    ```

* **Trace tests** -
    
    The trace tests run all the presets available in Coach, and compare their csv output to traces we extracted after
    verifying each preset works correctly. The requirements for passing these tests are quite strict - all the values
    in the csv file should match the golden csv file exactly. The trace tests can be run in parallel to shorten the
    testing time. To run the tests in parallel use the following command:
    
    ```
    python3 rl_coach/tests/trace_tests.py -prl
    ```
