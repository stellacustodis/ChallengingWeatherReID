def simple_scheduler(optimizer, epoch, learning_rate):
    lr = learning_rate
    milestones = [10, 18, 25, 33, 40, 45]
    for milestone in milestones:
        if epoch >= milestone:
            lr /= 10
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
