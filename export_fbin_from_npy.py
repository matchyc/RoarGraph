import numpy as np

imgs_10M = np.array([])
text_10M = np.array([])
dim = 512
for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]:
    # append np arrays 
    img_name = f'./data/laion-10M/img_emb_{i}.npy'
    one_img = np.load(img_name)
    # print(img_name)
    # convert to float32
    one_img = one_img.astype(np.float32) 
    dim = one_img.shape[1]
    imgs_10M = np.append(imgs_10M, one_img).astype(np.float32)
    text_name = f'./data/laion-10M/text_emb_{i}.npy'
    one_text = np.load(text_name)
    one_text = one_text.astype(np.float32)
    text_10M = np.append(text_10M, one_text).astype(np.float32)
    # print(one_text.shape)

imgs_10M = imgs_10M.reshape(-1, dim)
text_10M = text_10M.reshape(-1, dim)
# print(text_10M.shape)

f_img = open('./data/laion-10M/base.10M.fbin', 'wb')
f_txt = open('./data/laion-10M/query.train.10M.fbin', 'wb')



# save imgs_10M to f, write num points and dimension at first
npts, dim = imgs_10M.shape
f_img.write(np.array([npts, dim]).astype(np.uint32).tobytes())
imgs_10M.tofile(f_img)
f_img.close()


# save text_10M to f, write num points and dimension at first
npts, dim = text_10M.shape
f_txt.write(np.array([npts, dim]).astype(np.uint32).tobytes())
text_10M.tofile(f_txt)
f_txt.close()

