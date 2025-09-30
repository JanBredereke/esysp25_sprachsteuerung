import logging
import pathlib
import platform

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from pipelineConfig.PipelineConfigLoader import PipelineConfigLoader


class TrainingDatabaseConnection:
    """
    Class for interacting with the training database through SQLAlchemy.
    """

    _MAC_SYSTEM_STRING = "Darwin"
    _LINUX_SYSTEM_STRING = "Linux"

    Base = declarative_base()

    def __init__(self, config: PipelineConfigLoader):
        """
        Initializes the database connection.
        """
        self._db_path = pathlib.Path(config.db_path)
        # Check if the folder exists
        if not self._db_path.parent.exists():
            self._db_path.parent.mkdir(parents=True)
        # Check if the path is absolute and whether we are on Mac or Linux
        if self._db_path.is_absolute() and (
            platform.system() == TrainingDatabaseConnection._MAC_SYSTEM_STRING
            or platform.system() == TrainingDatabaseConnection._LINUX_SYSTEM_STRING
        ):
            # Mac and unix require 4 leading slashes for absolute paths
            db_url = "sqlite:////" + str(self._db_path)
        else:
            # Use three leading slashes
            db_url = "sqlite:///" + str(self._db_path)
        # Use SQLite for now
        self._engine = create_engine(db_url)
        # Log the connection
        logging.info("Connected to sample database: " + db_url)
        self._session_maker = sessionmaker(bind=self._engine)
        TrainingDatabaseConnection.Base.metadata.create_all(self._engine)

    @property
    def session(self):
        """
        Returns a session object.
        """
        return self._session_maker()
