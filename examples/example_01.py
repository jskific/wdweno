import wdweno

wdweno.wdweno(in_image='../images/fig_02_head.png', out_image='../images/out_2x_image.png',
                method='2x', scale_exp=1, beta=2, verbose=True)

wdweno.wdweno(in_image='../images/fig_02_head.png', out_image='../images/out_tensor_image.png',
                method='tensor', scale_exp=1, beta=2)

wdweno.wdweno(in_image='../images/fig_02_head.png', out_image='../images/out_free_image.png',
                method='free', scale_factor=2, beta=2)
