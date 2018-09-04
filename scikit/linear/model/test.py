import pipeline_invoke

# Make sure you are running in an environment similar to pipeline_conda_environment.yaml

import os

if __name__ == '__main__':
    with open('./pipeline_test_request.json', 'rb') as fh:
        request_binary = fh.read()
  
    response = pipeline_invoke.invoke(request_binary)
    print(response)
