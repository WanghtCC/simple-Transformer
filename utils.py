def load_pretrained_weights(model, checkpoint):
    print('Loading checkpoint ...')
    model_dict = model.state_dict()
    checkpoint_dict = checkpoint['state_dict']
    checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)

def print_img(data):
    img = data.numpy().tolist()
    for i in img[0][0]:
        formatted_img = ''.join(['{:<4}'.format(item) for item in i])
        print(formatted_img)