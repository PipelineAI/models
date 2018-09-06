import pipeline_invoke


with open('./pipeline_test_request.json', 'rb') as fh:
  contents_binary = fh.read()

print(pipeline_invoke.invoke(contents_binary))
