package faygolover.rpsky;

import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.tags.TagKey;
import net.minecraft.world.level.biome.Biome;
import net.minecraftforge.fml.common.Mod;

/**
 * RP Sky — биом «Под плитами» ({@code rpsky:under_plates}, данные мода): вместо неба гигантский купол из бетона
 * и металла в дымке ({@code client/SkyRenderer}), без солнца, луны и звёзд. Другим биомам это небо дают тегом
 * {@link #PLATED_SKY} из датапака. Часть RP-семейства, RP Core не требует.
 */
@Mod(RpSky.MODID)
public class RpSky {
    public static final String MODID = "rpsky";

    /** Биомы с небом из плит. В моде — только «Под плитами»; добавить другие — датапак ({@code datapack/rpsky_biomes}). */
    public static final TagKey<Biome> PLATED_SKY = TagKey.create(Registries.BIOME, new ResourceLocation(MODID, "plated_sky"));
}
