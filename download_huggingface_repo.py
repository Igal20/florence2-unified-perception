from huggingface_hub import snapshot_download


def main():
    snapshot_download(repo_id="microsoft/Florence-2-large",  local_dir=r"C:\Users\IgalDmitriev\wsc\data\image_caption\models\Florence-2-large")


if __name__ == "__main__":
    main()


