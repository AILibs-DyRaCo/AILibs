import org.aeonbits.owner.Config.Sources;
import org.aeonbits.owner.Mutable;

@Sources({ "file:conf/performancesamplegenerator/samplegenerator.properties" })
public interface IPerformanceSampleGeneratorConfig extends Mutable{

	/**
	 * Number of repetitions for monte carlo cross validation.
	 */
	public static final String K_MCCV_REPEATS = "mccvRepeats";
	
	/**
	 * Seed for monte carlo cross validation.
	 */
	public static final String K_MCCV_SEED = "mccvSeed";
	
	/**
	 * Ratio of data used for training in monte carlo cross validation.
	 */
	public static final String K_MCCV_TRAIN_RATIO = "mccvTrainRatio";
	
	/**
	 * Seed for random search.
	 */
	public static final String K_RANDOM_SEARCH_SEED = "randomSearchSeed";

	/**
	 * OpenML Key.
	 */
	public static final String K_OPENML_KEY = "openMLKey";
	
	/**
	 * Database connection.
	 */
	public static final String DB_HOST = "db.host";
	public static final String DB_USER = "db.username";
	public static final String DB_PASS = "db.password";
	public static final String DB_NAME = "db.database";
	public static final String DB_TABLE = "db.table";
	public static final String DB_SSL = "db.ssl";
	
	
	
	@Key(K_MCCV_REPEATS)
	@DefaultValue("10")
	public int getNumberMCCVRepeats();

	@Key(K_MCCV_SEED)
	@DefaultValue("1")
	public int getMCCVSeed();

	@Key(K_MCCV_TRAIN_RATIO)
	@DefaultValue("0.7d")
	public double getMCCVTrainRatio();

	@Key(K_RANDOM_SEARCH_SEED)
	@DefaultValue("1")
	public int getRandomSearchSeed();
	
	@Key(K_OPENML_KEY)
	public String getOpenMLKey();
	
	@Key(DB_HOST)
	public String getDBHost();

	@Key(DB_USER)
	public String getDBUsername();

	@Key(DB_PASS)
	public String getDBPassword();

	@Key(DB_NAME)
	public String getDBDatabaseName();

	@Key(DB_TABLE)
	public String getDBTableName();

	@Key(DB_SSL)
	public Boolean getDBSSL();

}
