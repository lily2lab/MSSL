from data_provider import skeleton



def dataloader(train_data_paths, valid_data_paths, batch_size,
                  jointsdim, jointsnum, channel, input_length, sequence_length, is_training=True):
    '''
    Given a dataset name and returns a Dataset.
    Args:
        train_data_paths: List, [train_data_path1, train_data_path2...]
        valid_data_paths: List, [val_data_path1, val_data_path2...]
        batch_size: Int
        img_width: Int
        is_training: Bool
    Returns:
        if is_training:
            Two dataset instances for both training and evaluation.
        else:
            One dataset instance for evaluation.
    Raises:
        ValueError: If `dataset_name` is unknown.
    '''
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')

    print 'testing data :\n'
    test_input_param = {'paths': valid_data_list,
                        'minibatch_size': batch_size,
                        'input_data_type': 'float32',
                        'is_output_sequence': True,
                        'input_length':input_length,
                        'sequence_length':sequence_length,
                        'jointsnum':jointsnum,
                        'jointsdim':jointsdim,
                        'channel':channel,
                        'name': 'test iterator'}
    test_input_handle = skeleton.InputHandle(test_input_param)
    test_input_handle.begin(do_shuffle = False)
    print 'training data: \n'
    if is_training:
        train_input_param = {'paths': train_data_list,
                             'minibatch_size': batch_size,
                             'input_data_type': 'float32',
                             'is_output_sequence': True,
                             'input_length':input_length,
                             'sequence_length':sequence_length,
                             'jointsnum':jointsnum,
                             'jointsdim':jointsdim,
                             'channel':channel,
                             'name': 'train iterator'}
        train_input_handle = skeleton.InputHandle(train_input_param)
        train_input_handle.begin(do_shuffle = True)
        return train_input_handle, test_input_handle
    else:
        return test_input_handle
