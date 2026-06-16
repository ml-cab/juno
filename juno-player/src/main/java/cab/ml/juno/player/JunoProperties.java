package cab.ml.juno.player;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.MissingResourceException;
import java.util.Properties;
import java.util.logging.Logger;

import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.registry.ParallelismType;

public class JunoProperties extends Properties {

	private static final long serialVersionUID = JunoProperties.class.getName().hashCode();

	private static final Logger log = Logger.getLogger(JunoProperties.class.getName());

	public JunoProperties(Properties properties) {
		super(properties);
	}

	public static Properties loadProperties() {
		Properties props = new Properties();
		try (InputStream externalPropsIS = new FileInputStream("app.properties")) {
			props.load(externalPropsIS);
			log.info("External props are loaded \n" + props.toString().replace(",", "\n"));
		} catch (IOException ioe) {
			try (InputStream defaultPropsIS = JunoProperties.class.getClassLoader().getResource("app.properties")
					.openStream()) {
				props.load(defaultPropsIS);
				log.info("Default props are loaded \n" + props.toString().replace(",", "\n"));
			} catch (IOException e) {
				throw new MissingResourceException("Unable to load properties", JunoProperties.class.getName(),
						"app.properties");
			}
		}
		return props;
	};
	
	ParallelismType getPType() {
		return ParallelismType.valueOf(defaults.getProperty("cluster.p_type").toUpperCase());
	}

	ActivationDtype getDType() {
		return ActivationDtype.valueOf(defaults.getProperty("cluster.dtype").toUpperCase());
	}

	int getInt(String name) {
		return Integer.parseInt(defaults.getProperty(name));
	}

	boolean getBoolean(String name) {
		return Boolean.parseBoolean(defaults.getProperty(name));
	}

	double getDouble(String name) {
		return Double.parseDouble(defaults.getProperty(name));
	}

	float getFloat(String name) {
		return Float.parseFloat(defaults.getProperty(name));
	}
}
