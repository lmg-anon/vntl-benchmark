# Quick and dirty port of https://github.com/bodaay/HuggingFaceModelDownloader to Python.
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS

#    APPENDIX: How to apply the Apache License to your work.

#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.

#    Copyright [yyyy] [name of copyright owner]

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re
import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Lock
import shutil
from tqdm import tqdm
import argparse
import time

# Constants
AGREEMENT_MODEL_URL = "https://huggingface.co/%s"
AGREEMENT_DATASET_URL = "https://huggingface.co/datasets/%s"
RAW_MODEL_FILE_URL = "https://huggingface.co/%s/raw/%s/%s"
RAW_DATASET_FILE_URL = "https://huggingface.co/datasets/%s/raw/%s/%s"
LFS_MODEL_RESOLVER_URL = "https://huggingface.co/%s/resolve/%s/%s"
LFS_DATASET_RESOLVER_URL = "https://huggingface.co/datasets/%s/resolve/%s/%s"
JSON_MODELS_FILE_TREE_URL = "https://huggingface.co/api/models/%s/tree/%s/%s"
JSON_DATASET_FILE_TREE_URL = "https://huggingface.co/api/datasets/%s/tree/%s/%s"

# Global Variables
NUM_CONNECTIONS = 5
REQUIRES_AUTH = False
AUTH_TOKEN = ""

# Classes
class HfModel:
    def __init__(self, type, oid, size, path, local_size=0, needs_download=True, is_directory=False, is_lfs=False, appended_path="", skip_downloading=False, filter_skip=False, download_link="", lfs=None):
        self.type = type
        self.oid = oid
        self.size = size
        self.path = path
        self.local_size = local_size
        self.needs_download = needs_download
        self.is_directory = is_directory
        self.is_lfs = is_lfs
        self.appended_path = appended_path
        self.skip_downloading = skip_downloading
        self.filter_skip = filter_skip
        self.download_link = download_link
        self.lfs = HfLfs(**lfs) if lfs else None

class HfLfs:
    def __init__(self, oid, size, pointerSize):
        self.oid = oid
        self.size = size
        self.pointerSize = pointerSize

def download_model(model_dataset_name, append_filter_to_path, skip_sha, is_dataset, destination_base_path, destination_path, model_branch, concurrent_connections, token, silent_mode):
    global NUM_CONNECTIONS, REQUIRES_AUTH, AUTH_TOKEN
    NUM_CONNECTIONS = concurrent_connections

    model_p = model_dataset_name
    has_filter = False
    if ":" in model_p:
        model_p = model_dataset_name.split(":")[0]
        has_filter = True
    model_path = destination_path if destination_path else os.path.join(destination_base_path, model_p.replace("/", "_"))
    if token:
        REQUIRES_AUTH = True
        AUTH_TOKEN = token

    if has_filter and append_filter_to_path:
        filters = model_dataset_name.split(":")[1].split(",")
        for ff in filters:
            ffpath = f"{model_path}_f_{ff}"
            os.makedirs(ffpath, exist_ok=True)
            new_model_dataset_name = f"{model_p}:{ff}"
            err = process_hf_folder_tree(ffpath, is_dataset, skip_sha, new_model_dataset_name, model_branch, "", silent_mode)
            if err:
                return err
    else:
        os.makedirs(model_path, exist_ok=True)
        err = process_hf_folder_tree(model_path, is_dataset, skip_sha, model_dataset_name, model_branch, "", silent_mode)
        if err:
            return err
    return None

def process_hf_folder_tree(model_path, is_dataset, skip_sha, model_dataset_name, branch, folder_name, silent_mode):
    json_tree_variable = JSON_MODELS_FILE_TREE_URL
    raw_file_url = RAW_MODEL_FILE_URL
    lfs_resolver_url = LFS_MODEL_RESOLVER_URL
    agreement_url = AGREEMENT_MODEL_URL % model_dataset_name
    has_filter = False
    filter_bin_file_string = []

    if ":" in model_dataset_name and not is_dataset:
        has_filter = True
        f = model_dataset_name.split(":")
        model_dataset_name = f[0]
        filter_bin_file_string = f[1].lower().split(",")
        if not silent_mode:
            print(f"Filter has been applied, will include LFS Model Files that contains: {filter_bin_file_string}")

    if is_dataset:
        json_tree_variable = JSON_DATASET_FILE_TREE_URL
        raw_file_url = RAW_DATASET_FILE_URL
        lfs_resolver_url = LFS_DATASET_RESOLVER_URL
        agreement_url = AGREEMENT_DATASET_URL % model_dataset_name

    temp_folder = os.path.join(model_path, folder_name, "tmp")
    json_file_list_url = json_tree_variable % (model_dataset_name, branch, folder_name)
    json_files_list = fetch_file_list(json_file_list_url)

    # for file in json_files_list:
    #     file_path = os.path.join(model_path, file.path)
    #     if file.is_directory:
    #         os.makedirs(file_path, exist_ok=True)
    #         process_hf_folder_tree(model_path, is_dataset, skip_sha, model_dataset_name, branch, file.path, silent_mode)
    #     elif file.needs_download:
    #         print(repr(file.__dict__))
    #         if file.is_lfs or needs_download(file_path, file.size):
    #             download_file_multi_thread(temp_folder, file.download_link, file_path)
    #         else:
    #             download_single_threaded(file.download_link, file_path)

    gguf_repo = False
    for file in json_files_list:
        filename_lower_case = file.path.lower()
        if ".gguf" in filename_lower_case:
            gguf_repo = True
            break
    
    for file in json_files_list:
        file.appended_path = os.path.join(model_path, file.path)
        if file.type == "directory":
            file.is_directory = True
            os.makedirs(os.path.join(model_path, file.path), exist_ok=True)
            file.skip_downloading = True
            new_model_dataset_name = f"{model_dataset_name}:{','.join(filter_bin_file_string)}"
            process_hf_folder_tree(model_path, is_dataset, skip_sha, new_model_dataset_name, branch, file.path, silent_mode)
            if len(os.listdir(os.path.join(model_path, file.path))) == 0:
                os.rmdir(os.path.join(model_path, file.path))
            continue
        
        filename_lower_case = file.path.lower()
        if gguf_repo and "gguf" not in filename_lower_case:
            file.filter_skip = True
            continue

        file.download_link = raw_file_url % (model_dataset_name, branch, file.path)
        if file.lfs:
            file.is_lfs = True
            if has_filter:
                if filename_lower_case.endswith(".act") or filename_lower_case.endswith(".bin") or\
                    ".gguf" in filename_lower_case or\
                    filename_lower_case.endswith(".safetensors") or filename_lower_case.endswith(".pt") or filename_lower_case.endswith(".meta") or\
                    filename_lower_case.endswith(".zip") or filename_lower_case.endswith(".z01") or filename_lower_case.endswith(".onnx") or filename_lower_case.endswith(".data") or\
                    filename_lower_case.endswith(".onnx_data"):
                    file.filter_skip = True
                    for ff in filter_bin_file_string:
                        if ff.lower() in filename_lower_case:
                            file.filter_skip = False

    for file in json_files_list:
        if file.is_directory or file.filter_skip:
            continue
        filename = file.appended_path
        if os.path.exists(filename):
            file_info = os.stat(filename)
            size = file_info.st_size
            if not silent_mode:
                print(f"Checking existing file: {file.appended_path}")
            if size == file.size:
                file.skip_downloading = True
                if file.is_lfs:
                    if not skip_sha:
                        err = verify_checksum(file.appended_path, file.lfs.oid)
                        if err:
                            os.remove(file.appended_path)
                            file.skip_downloading = False
                            if not silent_mode:
                                print(f"Hash failed for LFS file: {file.appended_path}, will redownload/resume")
                            return err
                        if not silent_mode:
                            print(f"Hash matched for LFS file: {file.appended_path}")
                    else:
                        if not silent_mode:
                            print(f"Hash matching skipped for LFS file: {file.appended_path}")
                else:
                    if not silent_mode:
                        print(f"File size matched for non LFS file: {file.appended_path}")

        if file.lfs and not file.skip_downloading:
            resolver_url = lfs_resolver_url % (model_dataset_name, branch, file.path)
            get_link = get_redirect_link(resolver_url)
            if get_link:
                file.download_link = get_link

    for file in json_files_list:
        if file.is_directory:
            continue
        if file.skip_downloading:
            if not silent_mode:
                print(f"Skipping: {file.appended_path}")
            continue
        if file.filter_skip:
            if not silent_mode:
                print(f"Filter Skipping: {file.appended_path}")
            continue
        if not silent_mode:
            print(f"Downloading: {file.appended_path}")
        if file.is_lfs:
            download_file_multi_thread(temp_folder, file.download_link, file.appended_path)
            if not skip_sha:
                err = verify_checksum(file.appended_path, file.lfs.oid)
                if err:
                    os.remove(file.appended_path)
                    if not silent_mode:
                        print(f"Hash failed for LFS file: {file.appended_path}, will redownload/resume")
                    return err
                if not silent_mode:
                    print(f"Hash matched for LFS file: {file.appended_path}")
            else:
                if not silent_mode:
                    print(f"Hash matching skipped for LFS file: {file.appended_path}")
        else:
            download_single_threaded(file.download_link, file.appended_path)
            if os.path.exists(file.appended_path):
                file_info = os.stat(file.appended_path)
                size = file_info.st_size
                if size != file.size:
                    return f"File size mismatch: {file.appended_path}, filesize: {size}, needed size: {file.size}"

    return None

def fetch_file_list(json_file_list_url):
    response = requests.get(json_file_list_url)
    if response.status_code == 401 and not REQUIRES_AUTH:
        raise Exception("Repo requires access token, generate an access token from HuggingFace, and pass it using flag: -t TOKEN")
    if response.status_code == 403:
        raise Exception("You need to manually accept the agreement for this model/dataset on HuggingFace site, no bypass will be implemented")
    return [HfModel(**file) for file in response.json()]

def needs_download(file_path, remote_size):
    if not os.path.exists(file_path):
        return True
    return os.stat(file_path).st_size != remote_size

def is_valid_model_name(model_name):
    pattern = r'^[A-Za-z0-9_\-]+/[A-Za-z0-9\._\-]+$'
    return re.match(pattern, model_name) is not None

def download_single_threaded(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def download_file_multi_thread(temp_folder, url, output_path):
    os.makedirs(temp_folder, exist_ok=True)
    
    response = requests.head(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch headers for URL: {url}")

    file_size = int(response.headers.get('content-length', 0))
    if file_size == 0:
        raise Exception(f"Failed to retrieve content length for URL: {url}")

    part_size = file_size // NUM_CONNECTIONS
    futures = []

    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(output_path))
    lock = Lock()

    with ThreadPoolExecutor(max_workers=NUM_CONNECTIONS) as executor:
        for i in range(NUM_CONNECTIONS):
            start = i * part_size
            end = file_size if i == NUM_CONNECTIONS - 1 else start + part_size - 1
            part_path = os.path.join(temp_folder, f"part_{i}")
            futures.append(executor.submit(download_part, url, start, end, part_path, progress_bar, lock))

        not_done = futures
        try:
            while not_done:
                done, not_done = wait(not_done, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    response = future.result()  # Raise any exceptions that occurred during the download
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            progress_bar.close()
            #shutil.rmtree(temp_folder)
            os.kill(os.getpid(), 9)


    progress_bar.close()
    combine_parts(temp_folder, output_path)

    shutil.rmtree(temp_folder)

def download_part(url, start, end, output_path, progress_bar, lock):
    headers = {'Range': f'bytes={start}-{end}'}
    max_retries = 3
    delay = 1

    for attempt in range(max_retries):
        bytes_downloaded = 0
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    file.write(chunk)
                    chunk_size = len(chunk)
                    bytes_downloaded += chunk_size
                    with lock:
                        progress_bar.update(chunk_size)
            
            # Success, exit the function
            return
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.RequestException) as e:
            print(f"Error occurred (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Rewind the progress bar
            with lock:
                progress_bar.update(-bytes_downloaded)
            
            if attempt < max_retries - 1:
                time.sleep(delay)  # Wait before retrying


def combine_parts(temp_folder, output_path):
    with open(output_path, 'wb') as output_file:
        for part_file in sorted(os.listdir(temp_folder)):
            part_path = os.path.join(temp_folder, part_file)
            with open(part_path, 'rb') as part:
                shutil.copyfileobj(part, output_file)

def verify_checksum(file_path, sha):
    with open(file_path, 'rb') as file:
        file_hash = hashlib.sha256(file.read()).hexdigest()
    if file_hash != sha:
        return f"File hash mismatch: {file_path}, expected: {sha}, got: {file_hash}"
    return None

def get_redirect_link(url):
    response = requests.head(url, allow_redirects=True)
    return response.url if response.status_code == 200 else None

# Main function to download the model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default=None, nargs='?')
    parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
    parser.add_argument('--is-dataset', action='store_true', help='Specifies that this is a dataset.')
    parser.add_argument('--token', type=str, default=None, help='The huggingface token.')
    parser.add_argument('--threads', type=int, default=4, help='Number of simultaneous connections.')
    parser.add_argument('--output-base', type=str, default=".", help='The folder where the model folder should be saved.')
    parser.add_argument('--output', type=str, default=None, help='The folder where the model should be saved.')
    parser.add_argument('--check', action='store_true', help='Validates the checksums of model files.')
    parser.add_argument('--silent', action='store_true', help='Does not print to the console.')
    args = parser.parse_args()

    if args.model is None:
        print("No model specified.")
        return

    download_model(args.model, False, not args.check, args.is_dataset, args.output_base, args.output, args.branch, args.threads, args.token, args.silent)

if __name__ == "__main__":
    main()