package faygolover.rpsky;

import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.tags.TagKey;
import net.minecraft.world.level.biome.Biome;
import net.minecraftforge.fml.common.Mod;

/**
 * RP Sky — в выбранных биомах вместо неба тусклая серая дымка, за которой видны плиты из бетона и металла
 * ({@code client/SkyRenderer}). Биомы — тег {@link #PLATED_SKY} из датапака сервера: клиент получает теги биомов
 * при входе, так что мод на сервере не обязателен. Часть RP-семейства, RP Core не требует.
 */
@Mod(RpSky.MODID)
public class RpSky {
    public static final String MODID = "rpsky";

    /** Биомы с небом из плит. В самом моде пуст — список задаёт датапак ({@code datapack/rpsky_biomes}). */
    public static final TagKey<Biome> PLATED_SKY = TagKey.create(Registries.BIOME, new ResourceLocation(MODID, "plated_sky"));
}
