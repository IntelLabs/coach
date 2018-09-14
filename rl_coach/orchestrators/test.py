from rl_coach.orchestrators.kubernetes_orchestrator import KubernetesParameters, Kubernetes

# image = 'gcr.io/constant-cubist-173123/coach:latest'
image = 'ajaysudh/testing:coach'
command = ['python3', 'rl_coach/rollout_worker.py', '-p', 'CartPole_DQN_distributed']
# command = ['sleep', '10h']

params = KubernetesParameters(image, command, kubeconfig='~/.kube/config', redis_ip='redis-service.ajay.svc', redis_port=6379, num_workers=1)
# params = KubernetesParameters(image, command, kubeconfig='~/.kube/config')

obj = Kubernetes(params)
if not obj.setup():
    print("Could not setup")

if obj.deploy():
    print("Successfully deployed")
else:
    print("Could not deploy")
