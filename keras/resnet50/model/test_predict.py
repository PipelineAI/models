import pipeline_invoke

if __name__ == '__main__':
    with open('data/test_request.json', 'rb') as fh:
        request_binary = fh.read()
   
    response = pipeline_invoke.invoke(request_binary)
    print(response)

