// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MatchRegistry {
    struct MatchRecord {
        bytes32 artifactHash;
        bytes32 metadataHash;
        string sourceUrl;
        uint256 timestamp;
        address submitter;
    }

    uint256 public nextId = 1;
    mapping(uint256 => MatchRecord) private records;

    event MatchRecorded(uint256 indexed recordId, bytes32 artifactHash, bytes32 metadataHash, string sourceUrl);

    function recordMatch(bytes32 artifactHash, bytes32 metadataHash, string calldata sourceUrl) external returns (uint256 recordId) {
        recordId = nextId++;
        records[recordId] = MatchRecord(artifactHash, metadataHash, sourceUrl, block.timestamp, msg.sender);
        emit MatchRecorded(recordId, artifactHash, metadataHash, sourceUrl);
    }

    function getMatch(uint256 recordId) external view returns (bytes32, bytes32, string memory, uint256, address) {
        MatchRecord memory record = records[recordId];
        require(record.timestamp != 0, "record does not exist");
        return (record.artifactHash, record.metadataHash, record.sourceUrl, record.timestamp, record.submitter);
    }
}