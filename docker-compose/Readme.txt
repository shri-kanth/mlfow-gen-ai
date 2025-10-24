# Folder Structure Documentation

Before running this code, please ensure you have created the following folder structure in your project directory. This structure is required for proper functioning of the code and associated services.

- `data/minio`: Directory for MinIO data storage.
- `data/pgdb`: Directory for PostgreSQL database data.
- `docker-compose.yml`: Main Docker Compose configuration file.
- `gateway-config.yaml`: Configuration file for the gateway service.
- `.env` : File containing environment variables and secrets, use `.env.template` as reference  

Make sure all directories and files are present as shown to avoid runtime errors.
├── data
│   ├── minio
│   └── pgdb
├── docker-compose.yml
└── gateway-config.yaml
└── .env
