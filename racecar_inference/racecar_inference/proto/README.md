# Install protobuf
Follow instructions here https://github.com/protocolbuffers/protobuf  
had some issue installing from apt before  
it goes something like (get the right release string):
```
mkdir protoc && cd protoc
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.19.6/protoc-3.19.6-linux-aarch_64.zip
unzip protoc-3.19.6-linux-aarch_64.zip
sudo mv bin/protoc /usr/local/bin
sudo mv include/* /usr/local/include
cd ../ && rm -rf protoc
```
note about the version: i choose version 3.19.6 bc it's the latest to support python3.6
I need to escape from python3.6 for real.


# Compile protobuf files
`protoc -I=./ --python_out=./ ./protocol.proto`
