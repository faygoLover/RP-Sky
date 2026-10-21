package faygolover.rpsky.client;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.BufferBuilder;
import com.mojang.blaze3d.vertex.DefaultVertexFormat;
import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.Tesselator;
import com.mojang.blaze3d.vertex.VertexFormat;
import faygolover.rpsky.RpSky;
import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.DimensionSpecialEffects;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.util.Mth;
import net.minecraft.world.level.material.FogType;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.client.event.RenderLevelStageEvent;
import net.minecraftforge.client.event.ViewportEvent;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import org.joml.Matrix4f;

/**
 * Небо из плит: куб из шести текстур поверх ванильного неба (солнце, луна и звёзды под ним не видны) и туман
 * в цвет дымки. На границе биома небо проявляется и уходит за {@link #FADE_TICKS}; ночью и в дождь темнеет.
 */
@Mod.EventBusSubscriber(modid = RpSky.MODID, value = Dist.CLIENT)
public final class SkyRenderer {
    private SkyRenderer() {}

    /** За сколько тиков небо полностью проявляется или уходит. */
    private static final int FADE_TICKS = 60;
    /** Цвет дымки у горизонта днём — туман мира подстраивается под него. */
    private static final float HAZE_R = 0.62f, HAZE_G = 0.62f, HAZE_B = 0.61f;
    /** Яркость неба глубокой ночью (днём — 1). */
    private static final float NIGHT = 0.06f;

    /**
     * Грани куба: имя текстуры, центр, «вправо» (u), «вниз» (v). Так же описаны в {@code scripts/gen_sky.py} —
     * менять только вместе.
     */
    private static final Face[] FACES = {
            new Face("up", 0, 1, 0, 1, 0, 0, 0, 0, 1),
            new Face("down", 0, -1, 0, 1, 0, 0, 0, 0, -1),
            new Face("north", 0, 0, -1, -1, 0, 0, 0, -1, 0),
            new Face("south", 0, 0, 1, 1, 0, 0, 0, -1, 0),
            new Face("east", 1, 0, 0, 0, 0, -1, 0, -1, 0),
            new Face("west", -1, 0, 0, 0, 0, 1, 0, -1, 0),
    };

    private static float blend;
    private static float prevBlend;

    private record Face(ResourceLocation tex, float cx, float cy, float cz, float rx, float ry, float rz, float dx, float dy, float dz) {
        Face(String name, float cx, float cy, float cz, float rx, float ry, float rz, float dx, float dy, float dz) {
            this(new ResourceLocation(RpSky.MODID, "textures/sky/" + name + ".png"), cx, cy, cz, rx, ry, rz, dx, dy, dz);
        }

        /** Угол грани (u, v ∈ {0, 1}) на расстоянии 100. */
        void vertex(BufferBuilder b, Matrix4f m, int u, int v, float c, float a) {
            float su = 2 * u - 1, sv = 2 * v - 1;
            b.vertex(m, 100 * (cx + su * rx + sv * dx), 100 * (cy + su * ry + sv * dy), 100 * (cz + su * rz + sv * dz))
                    .uv(u, v).color(c, c, c, a).endVertex();
        }
    }

    @SubscribeEvent
    public static void onTick(TickEvent.ClientTickEvent e) {
        if (e.phase != TickEvent.Phase.END) return;
        Minecraft mc = Minecraft.getInstance();
        prevBlend = blend;
        if (mc.level == null || mc.player == null) {
            blend = prevBlend = 0;
            return;
        }
        boolean in = mc.level.getBiome(mc.player.blockPosition()).is(RpSky.PLATED_SKY);
        blend = Mth.clamp(blend + (in ? 1f : -1f) / FADE_TICKS, 0, 1);
    }

    private static float blend(float partialTick) {
        return Mth.lerp(partialTick, prevBlend, blend);
    }

    /** Можно ли рисовать: обычное небо измерения и камера не в жидкости и не в снегу. */
    private static boolean visible(ClientLevel level, Camera camera) {
        return level.effects().skyType() == DimensionSpecialEffects.SkyType.NORMAL && camera.getFluidInCamera() == FogType.NONE;
    }

    /** Яркость неба: день — 1, ночь — {@link #NIGHT}; дождь и гроза приглушают. */
    private static float brightness(ClientLevel level, float partialTick) {
        float day = Mth.clamp(Mth.cos(level.getTimeOfDay(partialTick) * Mth.TWO_PI) * 2 + 0.5f, 0, 1);
        float b = Mth.lerp(day, NIGHT, 1f);
        b *= 1 - level.getRainLevel(partialTick) * 0.2f;
        b *= 1 - level.getThunderLevel(partialTick) * 0.2f;
        return b;
    }

    @SubscribeEvent
    public static void onRender(RenderLevelStageEvent e) {
        if (e.getStage() != RenderLevelStageEvent.Stage.AFTER_SKY) return;
        Minecraft mc = Minecraft.getInstance();
        ClientLevel level = mc.level;
        float pt = e.getPartialTick();
        float a = blend(pt);
        if (level == null || a <= 0.001f || !visible(level, e.getCamera())) return;
        float c = brightness(level, pt);

        PoseStack pose = e.getPoseStack();
        Matrix4f m = pose.last().pose();
        RenderSystem.enableBlend();
        RenderSystem.defaultBlendFunc();
        RenderSystem.depthMask(false);
        RenderSystem.disableCull();
        RenderSystem.setShader(GameRenderer::getPositionTexColorShader);
        RenderSystem.setShaderColor(1, 1, 1, 1);
        Tesselator tesselator = Tesselator.getInstance();
        BufferBuilder b = tesselator.getBuilder();
        for (Face f : FACES) {
            RenderSystem.setShaderTexture(0, f.tex());
            b.begin(VertexFormat.Mode.QUADS, DefaultVertexFormat.POSITION_TEX_COLOR);
            f.vertex(b, m, 0, 0, c, a);
            f.vertex(b, m, 0, 1, c, a);
            f.vertex(b, m, 1, 1, c, a);
            f.vertex(b, m, 1, 0, c, a);
            tesselator.end();
        }
        RenderSystem.enableCull();
        RenderSystem.depthMask(true);
        RenderSystem.disableBlend();
    }

    /** Туман мира — в цвет дымки, чтобы даль сливалась с небом. */
    @SubscribeEvent
    public static void onFogColor(ViewportEvent.ComputeFogColor e) {
        Minecraft mc = Minecraft.getInstance();
        ClientLevel level = mc.level;
        float pt = (float) e.getPartialTick();
        float a = blend(pt);
        if (level == null || a <= 0.001f || !visible(level, e.getCamera())) return;
        float c = brightness(level, pt);
        e.setRed(Mth.lerp(a, e.getRed(), HAZE_R * c));
        e.setGreen(Mth.lerp(a, e.getGreen(), HAZE_G * c));
        e.setBlue(Mth.lerp(a, e.getBlue(), HAZE_B * c));
    }
}
