docker context rm DFP_development
docker context create DFP_development --docker "host=ssh://theo.vincent@$1"
docker context use DFP_development

docker run -d -it -v /home/theo.vincent:/home/developer/ gcr.io/directfutureprediction/development_vm:first_try bash