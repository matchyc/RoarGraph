# get argument "dataset", each dataset has a code branch

# check if the dataset is provided
if [ -z "$1" ]; then
    echo "Please provide the dataset name with t2i-10M | laion-10M | clip-webvid-2.5M"
    exit 1
fi

# check if the dataset is valid
if [ "$1" != "t2i-10M" ] && [ "$1" != "laion-10M" ] && [ "$1" != "clip-webvid-2.5M" ]; then
    echo "Invalid dataset name in [t2i-10M, laion-10M, clip-webvid-2.5M]"
    exit 1
fi

dataset=$1

mkdir -p data
mkdir -p data/$1

if [ "$1" == "t2i-10M" ]; then
    echo "dataset t2i"
    need_size=$((200*4*10000000+8-1))
    query_10k_size=$((200*4*10000+8-1))
    # download the dataset
    if [ ! -e ./data/$1/gt.10k.ibin ]; then
        curl -r 0-${need_size} -o data/$1/base.10M.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.10M.fbin
        curl -r 0-${need_size} -o data/$1/query.train.10M.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
        curl -r 0-${query_10k_size} -o data/$1/query.10k.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
        curl -o data/$1/gt.10k.ibin https://zenodo.org/records/11090378/files/t2i.gt.10k.ibin
    fi
    curl -o data/$1/gt.10k.ibin https://zenodo.org/records/11090378/files/t2i.gt.10k.ibin
    python3 change_meta_data_in_file.py ./data/t2i-10M/query.train.10M.fbin 10000000
    python change_meta_data_in_file.py ./data/t2i-10M/query.10k.fbin 10000
elif [ "$1" == "laion-10M" ]; then
    echo "dataset laion"
    # download the dataset
    for i in 0 1 2 3 4 5 6 7 9 10
    do
        if [ ! -e ./data/$1/img_emb_${i}.npy ]; then
            wget -t 0 https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images/img_emb_${i}.npy -P data/$1
        fi
    done

    for i in 0 1 2 3 4 5 6 7 9 10
    do
        if [ ! -e ./data/$1/text_emb_${i}.npy ]; then
            wget -t 0 https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/texts/text_emb_${i}.npy -P data/$1
        fi
    done

    # export text and img simultaneously, watch out the DRAM.
    python3 export_fbin_from_npy.py
    if [ ! -e ./data/$1/gt.10k.ibin ]; then
        curl -o data/$1/query.10k.fbin https://zenodo.org/records/11090378/files/laion.query.10k.fbin
        curl -o data/$1/gt.10k.ibin https://zenodo.org/records/11090378/files/laion.gt.10k.ibin
    fi
elif [ "$1" == "clip-webvid-2.5M" ]; then
    echo "dataset clip-webvid"
    if [ ! -e ./data/clip-webvid-2.5M/base.2.5M.fbin ]; then
        wget -O ./data/clip-webvid-2.5M/base.2.5M.fbin https://zenodo.org/records/11090378/files/clip.webvid.base.2.5M.fbin
        # you can run prepare_for_clip_webvid on your own to generate base.2.5M.fbin.
        # mkdir -p ./data/clip-webvid-2.5M/temp_tar_data/
        # python3 prepare_for_clip_webvid.py
    fi

    if [ ! -e ./data/clip-webvid-2.5M/query.train.2.5M.fbin ]; then
        curl -o data/clip-webvid-2.5M/query.train.2.5M.fbin https://zenodo.org/records/11090378/files/webvid.query.train.2.5M.fbin
    fi

    if [ ! -e ./data/clip-webvid-2.5M/gt.10k.ibin ]; then
        curl -o data/clip-webvid-2.5M/query.10k.fbin https://zenodo.org/records/11090378/files/webvid.query.10k.fbin
        curl -o data/clip-webvid-2.5M/gt.10k.ibin https://zenodo.org/records/11090378/files/webvid.gt.10k.ibin
    fi
fi
