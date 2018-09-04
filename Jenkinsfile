/*
* NervanaSystems/private-coach Jenkinsfile
*/

// Constant Variables for Build
// To define the node label - jenkins -> manage jenkins -> manage nodes -> select requested node -> labels
static final String nodeLabel       = 'gpu'
static final String customWorkspace = '/state/ws'
static final Map slackColorMap      = [
    'FAILURE': 'danger',
    'UNSTABLE': 'warning',
    'SUCCESS': 'good'
]
static final Map ansiColorMap       = [
    'FAILURE': '\u001B[31m',
    'WARNING': '\u001B[33m',
    'SUCCESS': '\u001B[32m',
    'END': '\u001B[0m'
]

// Common Closures for Build
def slackStartMessage = {
    try {
        slackSend message: "Build ${env.JOB_NAME} started (<${env.BUILD_URL}|LINK>)"
    } catch (err) {
        echo "BUILD WARNING - Failed to send Slack Message: ${err}"
    }
}
def slackEndMessage = {
    try {
        slackSend color: slackColorMap[currentBuild.currentResult], message: "Build ${env.JOB_NAME} finished with result: ${currentBuild.currentResult} (<${env.BUILD_URL}|LINK>)"
    } catch (err) {
        echo "BUILD WARNING - Failed to send Slack Message: ${err}"
    }
}


// Wrap entire build with timestamps to improve Jenkins console log readability
timestamps {
    node(nodeLabel) {

        // Send Slack start message
        slackStartMessage()

        try {
            ansiColor('xterm') {
                // Clear previous workspace
                deleteDir()

                stage('Checkout') {
                    // Clone repo at triggered commit
                    checkout scm
                }

                stage('Build') {
                    // Build docker image
                    sh 'docker build -t coach --build-arg http_proxy="http://proxy-chain.intel.com:911" --build-arg https_proxy="http://proxy-chain.intel.com:912" -f Dockerfile .'
                }

                stage('Test') {
                    // Unit tests - short and contained functionality tests that take up to 1 minute.
                    stage('Unit Tests') {
                        try {
                            sh 'docker run\
                                -e http_proxy="http://proxy-chain.intel.com:911"\
                                -e https_proxy="http://proxy-chain.intel.com:912"\
                                -e MUJOCO_KEY=\$MUJOCO_KEY coach pytest rl_coach/tests -m unit_test'
                        } catch (err) {
                            echo "${ansiColorMap['FAILURE']} BUILD FAILURE - Caught Exception: ${err} ${ansiColorMap['END']}"
                            currentBuild.result = 'FAILURE'
                        }
                    }

                    // Integration tests - long functionality tests which can take up to 1 hour.
                    //stage('Integration Tests') {
                    //    try {
                    //        sh 'docker run\
                    //              -e http_proxy="http://proxy-chain.intel.com:911"\
                    //              -e https_proxy="http://proxy-chain.intel.com:912"\
                    //              -e MUJOCO_KEY=\$MUJOCO_KEY coach pytest rl_coach/tests -m integration_test -s'
                    //    } catch (err) {
                    //        echo "${ansiColorMap['FAILURE']} BUILD FAILURE - Caught Exception: ${err} ${ansiColorMap['END']}"
                    //        currentBuild.result = 'FAILURE'
                    //    }
                    //}

                    // Trace tests - long tests which test for equality to known output
                    stage('Trace Tests') {
                        try {
                            sh 'docker run\
                                  -e http_proxy="http://proxy-chain.intel.com:911"\
                                  -e https_proxy="http://proxy-chain.intel.com:912"\
                                  -e MUJOCO_KEY=\$MUJOCO_KEY coach python3 rl_coach/tests/trace_tests.py -prl\
                                  -ip Doom_Basic_BC,MontezumaRevenge_BC,Carla_3_Cameras_DDPG,Carla_DDPG,Carla_Dueling_DDQN,Starcraft_CollectMinerals_A3C,Starcraft_CollectMinerals_Dueling_DDQN'
                        } catch (err) {
                            echo "${ansiColorMap['FAILURE']} BUILD FAILURE - Caught Exception: ${err} ${ansiColorMap['END']}"
                            currentBuild.result = 'FAILURE'
                        }
                    }

                    // Golden tests - long tests which test for performance in terms of score and sample efficiency
                    stage('Golden Tests') {
                        try {
                            sh 'docker run\
                                  -e http_proxy="http://proxy-chain.intel.com:911"\
                                  -e https_proxy="http://proxy-chain.intel.com:912"\
                                  -e MUJOCO_KEY=\$MUJOCO_KEY coach python3 rl_coach/tests/golden_tests.py -np'
                        } catch (err) {
                            echo "${ansiColorMap['FAILURE']} BUILD FAILURE - Caught Exception: ${err} ${ansiColorMap['END']}"
                            currentBuild.result = 'FAILURE'
                        }
                    }

                    // TODO: run PEP8 style checks?
                }
            }
        } catch (err) {
            fail(${err})
        }

        // Send Slack end message
        slackEndMessage()

    }
}