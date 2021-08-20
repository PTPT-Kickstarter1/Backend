const express = require("express");

const server = express();
server.use(express.json());

server.get("/", (req, res) => {
  res.json({ message: "Homepage" });
});

server.get("/api", (req, res) => {
  res.json({ message: "api is working" });
});

module.exports = server;
