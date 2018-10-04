import os
import json

import logging
import tarfile
import argparse
import shutil

logging.basicConfig(format='%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s',
                        level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Path to json file')
    parser.add_argument('--destination', type=str, help='Path to folder with files in brat format')
    parser.add_argument('--ann-conf', type=str, help='Path to annotation.conf')

    args = parser.parse_args()

    with open(args.source, "r") as file:
        data = json.load(file)

    output_directory = args.destination.strip()

    if os.path.exists(output_directory):
        logger.warning("Directory already exists and will be deleted.")
        shutil.rmtree(output_directory)

    os.mkdir(output_directory)
    shutil.copy(args.ann_conf, output_directory)

    for index, text_entities in enumerate(data):
        text, named_entities = text_entities['text'], text_entities['named_entities']
        flatten = []
        for key, value in named_entities.items():
            for v in value:
                flatten.append((key, v))

        flatten = sorted(flatten, key=lambda t: t[1][0])
        with open("%s/text_%d.ann" % (output_directory, index), "a", encoding='utf-8') as file:
            for i, row in enumerate(flatten):
                name, interval = row
                start, end = interval
                file.write("T{0}\t{1} {2} {3}\t{4}\n".format(i, name, start, end, text[start:end]))

        with open("%s/text_%d.txt" % (output_directory, index), "a", encoding='utf-8') as file:
            file.write(text)

    logger.info("Packing files into %s.tar.gz" % output_directory)
    with tarfile.open("%s.tar.gz" % output_directory, "w:gz") as tar:
        tar.add(output_directory)

    logger.info("Delete source folder %s" % output_directory)
    shutil.rmtree(output_directory)