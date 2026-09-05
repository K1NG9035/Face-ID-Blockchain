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
    mapping(bytes32 => uint256) public artifactToRecordId;

    event MatchRecorded(uint256 indexed recordId, bytes32 indexed artifactHash, bytes32 metadataHash, string sourceUrl);

    function recordMatch(bytes32 artifactHash, bytes32 metadataHash, string calldata sourceUrl) external returns (uint256 recordId) {
        recordId = nextId++;
        records[recordId] = MatchRecord(artifactHash, metadataHash, sourceUrl, block.timestamp, msg.sender);
        artifactToRecordId[artifactHash] = recordId;
        emit MatchRecorded(recordId, artifactHash, metadataHash, sourceUrl);
    }

    function getMatch(uint256 recordId) external view returns (bytes32, bytes32, string memory, uint256, address) {
        MatchRecord memory record = records[recordId];
        require(record.timestamp != 0, "record does not exist");
        return (record.artifactHash, record.metadataHash, record.sourceUrl, record.timestamp, record.submitter);
    }

    function getMatchByArtifact(bytes32 artifactHash) external view returns (uint256 recordId, bytes32 metadataHash, string memory sourceUrl, uint256 timestamp, address submitter) {
        recordId = artifactToRecordId[artifactHash];
        require(recordId != 0, "artifact not recorded");
        MatchRecord memory record = records[recordId];
        return (recordId, record.metadataHash, record.sourceUrl, record.timestamp, record.submitter);
    }

    function verifyArtifact(bytes32 artifactHash, bytes32 metadataHash) external view returns (bool isRecorded, bool isUntampered, uint256 recordId, uint256 timestamp) {
        recordId = artifactToRecordId[artifactHash];
        if (recordId == 0) {
            return (false, false, 0, 0);
        }
        MatchRecord memory record = records[recordId];
        isRecorded = true;
        isUntampered = (record.metadataHash == metadataHash);
        timestamp = record.timestamp;
    }
}