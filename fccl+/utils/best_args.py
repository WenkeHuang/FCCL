best_args = {
    'fl_digits': {
        'fccl': {
            -1: {
                'communication_epoch': 40,
                'local_lr': 0.001,
                'public_lr': 0.001,
                'local_epoch': 20,
                'public_epoch': 1,
                'public_batch_size': 256,
                'local_batch_size': 128,
                'off_diag_weight': 0.0051,

            },
        },

        'fcclplus': {
            -1: {
                'communication_epoch': 40,
                'local_lr': 0.001,
                'public_lr': 0.001,
                'local_epoch': 20,
                'public_epoch': 1,
                'public_batch_size': 256,
                'local_batch_size': 128,
                'temp': 0.02,
                'dis_power': 3,
                'local_dis_power': 3
            },
        },
    },
    'fl_office31': {

        'fccl': {
            -1: {
                'communication_epoch': 40,
                'local_lr': 0.001,
                'public_lr': 0.001,
                'local_epoch': 20,
                'public_epoch': 1,
                'public_batch_size': 256,
                'local_batch_size': 128,
                'off_diag_weight': 0.0051,
            },
        },

        'fcclplus': {
            -1: {
                'communication_epoch': 40,
                'local_lr': 0.001,
                'public_lr': 0.001,
                'local_epoch': 20,
                'public_epoch': 1,
                'public_batch_size': 256,
                'local_batch_size': 128,
                'temp': 0.02,
                'dis_power': 3,
                'local_dis_power': 3
            },
        },

    },
    'fl_officecaltech': {
        'fccl': {
            -1: {
                'communication_epoch': 40,
                'local_lr': 0.001,
                'public_lr': 0.001,
                'local_epoch': 20,
                'public_epoch': 1,
                'public_batch_size': 128,
                'local_batch_size': 128,
                'off_diag_weight': 0.0051,
            },
        },

        'fcclplus': {
            -1: {
                'communication_epoch': 40,
                'local_lr': 0.001,
                'public_lr': 0.001,
                'local_epoch': 20,
                'public_epoch': 1,
                'public_batch_size': 128,
                'local_batch_size': 128,
                'temp': 0.02,
                'dis_power': 3,
                'local_dis_power': 1
            },
        },

    },
    'fl_officehome': {

        'fccl': {
            -1: {
                'communication_epoch': 40,
                'local_lr': 0.001,
                'public_lr': 0.001,
                'local_epoch': 10,
                'public_epoch': 1,
                'public_batch_size': 256,
                'local_batch_size': 128,
                'new': 'hh',
                'off_diag_weight': 0.0051,
            },
        },
        'fcclplus': {
            -1: {
                'communication_epoch': 40,
                'local_lr': 0.001,
                'public_lr': 0.001,
                'local_epoch': 15,
                'public_epoch': 2,
                'public_batch_size': 176,
                'local_batch_size': 128,
                'temp': 0.02,
                'dis_power': 1,
                'local_dis_power': 1
            },
        },


    },
}
