import sys, os
import json
import argparse
import urllib.request
from getpass import getpass

# Your airhdl.com user name and passord
USERNAME = "pedro.gimenes19@imperial.ac.uk"

FILE_TYPES = {
    "cHeader": ("_regs", "h"),
    "cppHeader": ("_regs", "hpp"),
    "svPackage": ("_regs_pkg", "sv"),
    "svModule": ("_regs", "sv"),
    "svTestbench": ("_regs_tb", "sv"),
    "vhdlPackage": ("_regs_pkg", "vhd"),
    "vhdlComponent": ("_regs", "vhd"),
    "vhdlInstance": ("_regs", "vho"),
    "vhdlTestbench": ("_regs_tb", "vhd"),
    "htmlDoc": ("_regs", "html"),
    "mdDoc": ("_regs", "md"),
    "ipxact": ("_regs", "xml"),
    "json": ("_regs", "json"),
    "all": ("_regs", "zip")
}

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='airhdl.com file download')
    parser.add_argument('register_map_id',
                        help='ID of the airhdl register map')
    parser.add_argument(
        'file_type', help='type of the file to download (e.g. vhdlComponent)')
    args = parser.parse_args()
    register_map_id = args.register_map_id
    file_type = args.file_type

    if file_type not in FILE_TYPES.keys():
        print("ERROR: unknown file type " + file_type)
        sys.exit(-1)

    # Get authentication token
    url = "https://airhdl.com/api/auth/login"
    PASSWORD = os.environ[f"AIRHDL_PASSWORD"]
    body = {'username': USERNAME, "password": PASSWORD}
    req = urllib.request.Request(
        url=url, method="POST")
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    resp = urllib.request.urlopen(req, jsondataasbytes, timeout=5)
    assert resp.status == 200
    resp_str = resp.read().decode('utf-8')
    resp_json = json.loads(resp_str)
    token = resp_json["token"]

    # Get the register map name
    url = "https://airhdl.com/api/registermaps/" + register_map_id + "?includeRegisters=false"
    req = urllib.request.Request(
        url=url, method="GET")
    arg2 = "Bearer " + token
    req.add_header("authorization", arg2)
    resp = urllib.request.urlopen(req, timeout=5)
    assert resp.status == 200
    resp_str = resp.read().decode('utf-8')
    resp_json = json.loads(resp_str)
    name = resp_json["name"]

    # Download the requested file type for the given register map
    url = "https://airhdl.com/api/registermaps/" + register_map_id + "/download?type=" + file_type
    req = urllib.request.Request(
        url=url, method="GET")
    req.add_header("authorization", arg2)
    resp = urllib.request.urlopen(req, timeout=5)
    assert resp.status == 200
    resp_headers = resp.getheaders()
    resp_content_type = resp.getheader('Content-Type')

    # Save the downloaded file
    suffix = FILE_TYPES[file_type][0]
    extension = FILE_TYPES[file_type][1]
    filename = name + suffix + "." + extension
    with open(filename, mode="wb") as f:
        f.write(resp.read())
        print(filename)