## ------------- Builtin imports --------------------------------------------------------------------
import json
import logging

# ------------- 3rd-party imports ------------------------------------------------------------------
import tensorflow as tf
from tensorflow.contrib import predictor

# --- PipelineAI imports ---------------------------------------------------------------------------
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
import base64

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.DEBUG)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'name': 'mnist',
           'tag': 'raw',
           'runtime': 'python',
           'chip': 'cpu',
           'resource_type': 'model',
           'resource_subtype': 'tensorflow',
          }


def _initialize_upon_import():

    try:
        saved_model_path = './pipeline_tfserving/0'
        return predictor.from_saved_model(saved_model_path)
    except Exception:
        _logger.error('pipeline_invoke_python._initialize_upon_import.Exception:', exc_info=True)

    return None


_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    """
    Transform bytes posted to the api into a Tensor.
    Classify the image
    Transform the model prediction output from a 1D array to a list of classes and probabilities

    :param bytes request:   byte array containing the content required by the predict method

    :return:                Response obj serialized to a JSON formatted str
                                containing a list of classes and a list of probabilities
    """
    try:
        with monitor(labels=_labels, name="transform_request"):
            transformed_request = _transform_request(request)

        with monitor(labels=_labels, name="invoke"):
            response = _model(transformed_request)

        with monitor(labels=_labels, name="transform_response"):
            transformed_response = _transform_response(response)

        return transformed_response

    except Exception:
        _logger.error('pipeline_invoke_python.invoke.Exception:', exc_info=True)


def _transform_request(request):
    # Convert from "browser view":
    #     data:image/png;base64,iVBORw0KGgo
    # To a normal base64:
    #     iVBORw0KGgo
    #
    # (See http://freeonlinetools24.com/base64-image for more details)
    request = request.decode('utf-8')
    if (request.startswith('data:')):
#        request = request.split('base64,')[1]
        request = request.split('data:')[1]
        image_type = request.split(';')[0]
        request = request.split(';')[1].split('base64,')[1]
        request = base64.b64decode(request, validate=True)

    # TODO:  Replace this with tf.image.decode_image() once we move to ArrayBuffer in the UI
    #        decode_image() doesn't work with the current request format (data: transformed above) for some reason
    if image_type == 'image/png':
        request_image_tensor = tf.image.decode_png(request, channels=1)
    if image_type == 'image/jpg':
        request_image_tensor = tf.image.decode_jpeg(request, channels=1)
    if image_type == 'image/bmp':
        request_image_tensor = tf.image.decode_bmp(request, channels=1)
# TODO:  Handle gif differently per this doc: https://www.tensorflow.org/api_docs/python/tf/image/decode_image
#    if image_type == 'image/gif':
#        request_image_tensor = tf.image.decode_gif(request, channels=1)

    request_image_tensor_resized = tf.image.resize_images(request_image_tensor, size=[28, 28])

    sess = tf.Session()
    with sess.as_default():
        request_np = request_image_tensor_resized.eval()
        request_np = (request_np / 255.0).reshape(1, 28, 28)

    return {"image": request_np}


def _transform_response(response):

    return json.dumps({
        "classes": response['classes'].tolist(),
        "probabilities": response['probabilities'].tolist(),
    })


if __name__ == '__main__':
#    with open('./pipeline_test_request.png', 'rb') as fb:
#        request_bytes = fb.read()
#        response_json = invoke(request_bytes)
#        print(response_json)

   # This is how the browser sends the image...
    request_bytes = b'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAMFGlDQ1BJQ0MgUHJvZmlsZQAASImVVwdYU8kWnltSCAktEAEpoXekSJfeO9LBRkgChBIhIajYkUUF14KKBURFV0AUXAsgiw27sgj2/kBFZWVdLNhQeZMCur72vfN9c+fPmXPO/GfuuZMZABRtWLm52agSADn8fEF0oA8zMSmZSeoDBDABqAB14MZiC3O9o6LCAJSx/u/y7iZAxP01K3Gsfx3/r6LM4QrZACBREKdyhOwciA8DgKuzcwX5ABA6od5gTn6uGA9BrCqABAEg4mKcLsXqYpwqxZYSm9hoX4i9ACBTWSxBOgAKYt7MAnY6jKMg5mjD5/D4EFdB7MHOYHEgvg+xZU7ObIgVyRCbpn4XJ/1vMVPHY7JY6eNYmotEyH48YW42a97/uRz/W3KyRWNz6MNGzRAERYtzhutWlzU7VIypELfzUyMiIVaB+AKPI7EX47sZoqA4mf0gW+gL1wwwAEABh+UXCrEWxAxRVpy3DNuxBBJfaI9G8PKDY2U4VTA7WhYfLeBnR4TJ4qzI4AaP4Wqu0D9mzCaNFxAMMaw09HBhRmyClCd6poAXHwGxAsTdwqyYUJnvw8IM34gxG4EoWszZEOK3aYKAaKkNpp4jHMsLs2azJHPBWsC88jNig6S+WCJXmBg2xoHD9fOXcsA4XH6cjBsGq8snWuZbkpsdJbPHqrnZgdHSdcYOCAtixnyv5sMCk64D9iiTFRIlm+tdbn5UrJQbjoIw4Av8ABOIYEsFs0Em4HUNtgzCX9KRAMACApAOuMBKphnzSJCM8OEzBhSCPyHiAuG4n49klAsKoP7LuFb6tAJpktECiUcWeApxDq6Je+BueBh8esFmhzvjLmN+TMWxWYn+RD9iEDGAaDbOgw1ZZ8MmALx/owuFPRdmJ+bCH8vhWzzCU0IP4RHhBqGXcAfEgyeSKDKrWbwiwQ/MmSAc9MJoAbLsUr/PDjeGrB1wH9wd8ofccQauCazwyTATb9wT5uYAtd8zFI1z+7aWP84nZv19PjK9grmCg4xF6vib8R23+jGK73drxIF96I+W2ArsEHYeO4VdxNqxFsDETmCtWCd2TIzHK+GJpBLGZouWcMuCcXhjNjYNNgM2n3+YmyWbX7xewnzu3Hzxx+A7O3eegJeekc/0hrsxlxnMZ1tbMu1sbJ0AEO/t0q3jDUOyZyOMS990eScBcCmFyvRvOpYBAEefAkB/901n8BqW+1oAjnWzRYICqU68HcP/DApQhF+FBtABBsAU5mMHHIEb8AL+IAREgliQBGbCFc8AOZDzHLAALAUloAysBRvBVrAd7AJ1YD84CFpAOzgFzoHLoBvcAPdgXfSDF2AIvAMjCIKQEBpCRzQQXcQIsUDsEGfEA/FHwpBoJAlJQdIRPiJCFiDLkDKkHNmK7ETqkV+Ro8gp5CLSg9xB+pAB5DXyCcVQKqqKaqPG6CTUGfVGQ9FYdAaajuahhWgxuhrdjNag+9Bm9BR6Gb2B9qIv0GEMYPIYA9PDrDBnzBeLxJKxNEyALcJKsQqsBmvE2uB7vob1YoPYR5yI03EmbgVrMwiPw9l4Hr4IX4VvxevwZvwMfg3vw4fwrwQaQYtgQXAlBBMSCemEOYQSQgVhD+EI4Sz8bvoJ74hEIoNoQnSC32USMZM4n7iKuI3YRDxJ7CE+Jg6TSCQNkgXJnRRJYpHySSWkLaR9pBOkq6R+0geyPFmXbEcOICeT+eQicgV5L/k4+Sr5GXlETknOSM5VLlKOIzdPbo3cbrk2uSty/XIjFGWKCcWdEkvJpCylbKY0Us5S7lPeyMvL68u7yE+V58kvkd8sf0D+gnyf/EeqCtWc6kudThVRV1NrqSepd6hvaDSaMc2LlkzLp62m1dNO0x7SPijQFawVghU4CosVKhWaFa4qvFSUUzRS9FacqVioWKF4SPGK4qCSnJKxkq8SS2mRUqXSUaVbSsPKdGVb5UjlHOVVynuVLyo/VyGpGKv4q3BUilV2qZxWeUzH6AZ0Xzqbvoy+m36W3q9KVDVRDVbNVC1T3a/apTqkpqI2WS1eba5apdoxtV4GxjBmBDOyGWsYBxk3GZ8maE/wnsCdsHJC44SrE96rT1T3Uueql6o3qd9Q/6TB1PDXyNJYp9Gi8UAT1zTXnKo5R7Na86zm4ETViW4T2RNLJx6ceFcL1TLXitaar7VLq1NrWFtHO1A7V3uL9mntQR2GjpdOps4GneM6A7p0XQ9dnu4G3RO6fzDVmN7MbOZm5hnmkJ6WXpCeSG+nXpfeiL6Jfpx+kX6T/gMDioGzQZrBBoMOgyFDXcNwwwWGDYZ3jeSMnI0yjDYZnTd6b2xinGC83LjF+LmJukmwSaFJg8l9U5qpp2meaY3pdTOimbNZltk2s25z1NzBPMO80vyKBWrhaMGz2GbRY0mwdLHkW9ZY3rKiWnlbFVg1WPVZM6zDrIusW6xfTjKclDxp3aTzk77aONhk2+y2uWerYhtiW2TbZvvaztyObVdpd92eZh9gv9i+1f7VZIvJ3MnVk2870B3CHZY7dDh8cXRyFDg2Og44GTqlOFU53XJWdY5yXuV8wYXg4uOy2KXd5aOro2u+60HXv9ys3LLc9ro9n2IyhTtl95TH7vruLPed7r0eTI8Ujx0evZ56nizPGs9HXgZeHK89Xs+8zbwzvfd5v/Sx8RH4HPF57+vqu9D3pB/mF+hX6tflr+If57/V/2GAfkB6QEPAUKBD4PzAk0GEoNCgdUG3grWD2cH1wUMhTiELQ86EUkNjQreGPgozDxOEtYWj4SHh68PvRxhF8CNaIkFkcOT6yAdRJlF5Ub9NJU6Nmlo59Wm0bfSC6PMx9JhZMXtj3sX6xK6JvRdnGieK64hXjJ8eXx//PsEvoTyhN3FS4sLEy0maSbyk1mRScnzynuThaf7TNk7rn+4wvWT6zRkmM+bOuDhTc2b2zGOzFGexZh1KIaQkpOxN+cyKZNWwhlODU6tSh9i+7E3sFxwvzgbOANedW859luaeVp72PN09fX36QIZnRkXGIM+Xt5X3KjMoc3vm+6zIrNqs0eyE7KYcck5KzlG+Cj+Lf2a2zuy5s3tyLXJLcnvzXPM25g0JQgV7hIhwhrA1XxUeczpFpqKfRH0FHgWVBR/mxM85NFd5Ln9u5zzzeSvnPSsMKPxlPj6fPb9jgd6CpQv6Fnov3LkIWZS6qGOxweLixf1LApfULaUszVr6e5FNUXnR22UJy9qKtYuXFD/+KfCnhhKFEkHJreVuy7evwFfwVnSttF+5ZeXXUk7ppTKbsoqyz6vYqy79bPvz5p9HV6et7lrjuKZ6LXEtf+3NdZ7r6sqVywvLH68PX9+8gbmhdMPbjbM2XqyYXLF9E2WTaFPv5rDNrVsMt6zd8nlrxtYblT6VTVVaVSur3m/jbLta7VXduF17e9n2Tzt4O27vDNzZXGNcU7GLuKtg19Pd8bvP/+L8S/0ezT1le77U8mt766LrztQ71dfv1dq7pgFtEDUM7Ju+r3u/3/7WRqvGnU2MprID4IDowB+/pvx682DowY5DzocaDxsdrjpCP1LajDTPax5qyWjpbU1q7TkacrSjza3tyG/Wv9W267VXHlM7tuY45Xjx8dEThSeGT+aeHDyVfupxx6yOe6cTT18/M/VM19nQsxfOBZw7fd77/IkL7hfaL7pePHrJ+VLLZcfLzZ0OnUd+d/j9SJdjV/MVpyut3S7dbT1Teo5f9bx66prftXPXg69fvhFxo+dm3M3bt6bf6r3Nuf38TvadV3cL7o7cW3KfcL/0gdKDiodaD2v+YfaPpl7H3mN9fn2dj2Ie3XvMfvziifDJ5/7ip7SnFc90n9U/t3vePhAw0P3HtD/6X+S+GBks+VP5z6qXpi8P/+X1V+dQ4lD/K8Gr0der3mi8qX07+W3HcNTww3c570bel37Q+FD30fnj+U8Jn56NzPlM+rz5i9mXtq+hX++P5oyO5rIELMlRAIMNTUsD4HUtALQkeHboBoCiIL17SQSR3hclCPwnLL2fScQRgFovAOKWABAGzyjVsBlBTIW9+Ogd6wVQe/vxJhNhmr2dNBYV3mAIH0ZH32gDQGoD4ItgdHRk2+jol92Q7B0ATuZJ73xiIcLz/Y5JYtTd/xL8KP8Ep4BtRBcsVDMAAAAJcEhZcwAAFiUAABYlAUlSJPAAAAICaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA1LjQuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj43ODwvZXhpZjpQaXhlbFlEaW1lbnNpb24+CiAgICAgICAgIDxleGlmOlBpeGVsWERpbWVuc2lvbj44NDwvZXhpZjpQaXhlbFhEaW1lbnNpb24+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgrisibRAAAEGUlEQVRIDY1WyUpjURA9iXEWR9QYFCc0jojgRkXdK7gQRRQXigsRXLnxM/wA24W0K0GccABRHJaCoAs3ERXBMY3zPHZOSYV0Oi95BeHd3Fe3TtWp4T7L5ubmd11dHSoqKvD09ITv728YidVqxd3dHc7Pz5GdnY2IiIig+rRjsVgQExODnZ0deLBg+/j4EPs0cnFxYYTl3e/s7ERNTQ2mpqawsrLi3Q+2SE9Pl9fEstEDSmRkJGw2W0CPo6Ki8Pz8jOLiYvT29iInJ0e85bnk5GTc3t5yGVBon7YpXNtU6+vrCxqt7qlSWFgY+L6xsRHl5eV4eXnB4+OjqH1+foK/YMKzKlZd8KnR+u4x6vf3d9mqrq5GUlKSADCfZsTfZtBTSgfpbGtrQ2lpqdDO6JRGX+/NOBASUCMpLCxESkqKUHt0dASXyyX2NXozYNQJCahtYrfbERsbK5U8PT2N09NTxMfH4/X11SyW6HmLJtApFsvb25u8ys3NBavV7XZjcXFR9uiMOkT6qc8ni8iI6qCALOf7+3t0dXWhoKBADO3v7+Py8hKkWqMLDw8XZ7RACGg0RAwBaZAVSqmtrQUpJY1LS0tStQkJCXh4eBBaGQ0d8xU6wfwqA/rOEFC9pSJbgQYY2fb2tp6V0cZRR+nr60N9fb3keX19HcPDw9Lw/kVlCOi16lnQS9LEfGpuOEfZGqSbLcOB4HA4xIkczyRyu/9gfPw3yISvmAIkvQQlbaSRwuIZGhpCR0cHSkpKxCFGQ0cyMzPh9LQRhf99xRCQlGo09JIHWYWsVMrAwAB6enrgdDpxcnKCubk50W9tbf3J/c+I9sWStWEfMiptCQKyYuPi4iTSqqoqtLe3Iz8/XwppYmIC/f39WFhYEAboqJ71RzSMkIoEpehQt2fY0dLSgrKyMlRWVsr+1tYWBgcHRa+5uRmpqak4PDzE7u6u7LF19LbghiEgi4QXJ+Xs7Ew8dmQ40N3dLVcSpw5vjaysLIyNjcmThcPo1tbWMDMzIy3DucuJpBIUUCNbXl5GUVGRVCInDguIhplXRsuhzjWLamNjA6Ojo2Kf0WkdhASkAqdFdHQ0RkZGJH/stby8POlJ7VPSTqMHBweYnJzE7Oys0MlzvGVUzxQgDdFLThw28tXVFRoaGiR//Gxga+zt7eH4+Bguz8j75XGMQrr1glYgfRpSqgoEpZc0wlzx19TUJKOOk2d1ddVrPDExUfrRf8ypLT5DAlKJBUR6OOKY1/n5eW6LcOSlpaXJ/s3NzX85Uz19mgKkMiO9vr4WetmXOpzpCKn1H9IK4P/8B9DMIUaonxf+xgL997fpBWS1GX0mBjJkdo/51wHCMzb1gNWofWfWmFk92qYQyxPUT5C8YDkR1AGzxkLpMUJOLH7VE+svCzsH88NbRDQAAAAASUVORK5CYII='

    response_json = invoke(request_bytes)
