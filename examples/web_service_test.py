import json
import pdb
import requests


URL = 'http://localhost:6667/predictInterests/{}'


def main():
    response = requests.get(URL.format(1))
    print(json.dumps(json.loads(response.text), indent=4))


if __name__ == '__main__':
    main()
