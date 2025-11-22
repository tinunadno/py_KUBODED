import math
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image


class Point2:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Point3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_list(l: list[float]):
        return Point3(l[0], l[1], l[2])

    def div(self, val: float):
        self.x /= val
        self.y /= val
        self.z /= val

    def norm(self):
        len_ = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        ret = Point3(self.x / len_, self.y / len_, self.z / len_)
        return ret

    def __sub__(self, other):
        return Point3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Point3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float):
        return Point3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float):
        return Point3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Point3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def length(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def rotate(self, axis: int, angle: float):
        axis_v = [0, 0, 0]
        axis_v[axis] = 1
        axis_v = Point3.from_list(axis_v)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        term1 = self * cos_angle
        term2 = axis_v.cross(self) * sin_angle
        term3 = axis_v * self.dot(axis_v) * (1 - cos_angle)
        return term1 + term2 + term3


def p_mul(p1: Point3, p2: Point3) -> Point3:
    return Point3(p1.x * p2.x, p1.y * p2.y, p1.z * p2.z)


class Ray:
    def __init__(self, pos: Point3, rot: Point3):
        self.pos = pos
        self.rot = rot

    def get_point(self, t: float) -> Point3:
        return Point3(
            self.pos.x + self.rot.x * t,
            self.pos.y + self.rot.y * t,
            self.pos.z + self.rot.z * t
        )


class Object(ABC):
    @abstractmethod
    def get_cross_point(self, ray: Ray) -> Point3 | None:
        pass

    @abstractmethod
    def get_normal(self, p: Point3) -> Point3:
        pass

    @abstractmethod
    def map_to_xy(self, pnt: Point3) -> Point2:
        pass


class LightSource(ABC):
    @abstractmethod
    def get_intensity(self, direction: Point3) -> Point3:
        pass

    @abstractmethod
    def pos(self) -> Point3:
        pass


class SimpleLightSource(LightSource):
    def __init__(self, pos: Point3, strength: Point3):
        self.position = pos
        self.strength = strength

    def pos(self) -> Point3:
        return self.position

    def get_intensity(self, _: Point3) -> Point3:
        return self.strength


class LambertianLightSource(LightSource):
    def __init__(self, pos: Point3, color: Point3, normal: Point3):
        self.position = pos
        self.strength = color
        self.normal = normal

    def pos(self) -> Point3:
        return self.position

    def get_intensity(self, direction: Point3) -> Point3:
        direction_to_point = direction.norm() * (-1)
        cos_theta = max(0, self.normal.dot(direction_to_point))
        return self.strength * cos_theta


class Material:
    def __init__(self, diffuse: Point3 | str, specular: Point3, shininess: float, ambient: Point3 = None):
        self.texture = None
        if isinstance(diffuse, str):
            self.texture = np.array(Image.open(diffuse).convert('RGB'))
        else:
            self.diffuse = diffuse
            self.ambient = ambient if ambient else diffuse * 0.1
        self.specular = specular
        self.shininess = shininess

    def __get_texture_color__(self, pix_cord: Point2) -> Point3:
        height, width, channels = self.texture.shape
        x = max(min(int(pix_cord.x * width), width - 1), 0)
        y = max(min(int(pix_cord.y * height), height - 1), 0)
        return Point3(float(self.texture[y, x, 0]) / 255.,
                      float(self.texture[y, x, 1]) / 255.,
                      float(self.texture[y, x, 2]) / 255.)

    def get_diffuse_color(self, pix_cord: Point2) -> Point3:
        if self.texture is None:
            return self.diffuse
        return self.__get_texture_color__(pix_cord)

    def get_ambient_color(self, pix_cord: Point2) -> Point3:
        if self.texture is None:
            return self.ambient
        return self.__get_texture_color__(pix_cord) * .1


class SceneObject:
    def __init__(self, obj: Object, mat: Material):
        self.obj = obj
        self.mat = mat


def apply_blin_fong(pnt: Point3, normal: Point3, c_pos: Point3, light: LightSource, mat: Material,
                    shadow_factor: float, pix_cord: Point2) -> Point3:
    px, py, pz = pnt.x, pnt.y, pnt.z
    nx, ny, nz = normal.x, normal.y, normal.z
    cx, cy, cz = c_pos.x, c_pos.y, c_pos.z

    light_pos = light.pos()
    lx, ly, lz = light_pos.x - px, light_pos.y - py, light_pos.z - pz
    light_len = (lx ** 2 + ly ** 2 + lz ** 2) ** 0.5
    lx, ly, lz = lx / light_len, ly / light_len, lz / light_len

    cx, cy, cz = cx - px, cy - py, cz - pz
    cam_len = (cx ** 2 + cy ** 2 + cz ** 2) ** 0.5
    cx, cy, cz = cx / cam_len, cy / cam_len, cz / cam_len
    hx, hy, hz = lx + cx, ly + cy, lz + cz
    h_len = (hx ** 2 + hy ** 2 + hz ** 2) ** 0.5
    hx, hy, hz = hx / h_len, hy / h_len, hz / h_len

    light_intensity = light.get_intensity(Point3(lx, ly, lz))

    base_ambient = mat.get_ambient_color(pix_cord)

    ambient = Point3(
        base_ambient.x * light_intensity.x,
        base_ambient.y * light_intensity.y,
        base_ambient.z * light_intensity.z
    )

    base_diffusion = mat.get_diffuse_color(pix_cord)

    diff = max(nx * lx + ny * ly + nz * lz, 0.)
    diffusion = Point3(
        base_diffusion.x * light_intensity.x * diff * shadow_factor,
        base_diffusion.y * light_intensity.y * diff * shadow_factor,
        base_diffusion.z * light_intensity.z * diff * shadow_factor
    )

    spec = max(nx * hx + ny * hy + nz * hz, 0.) ** mat.shininess
    specular = Point3(
        mat.specular.x * light_intensity.x * spec * shadow_factor,
        mat.specular.y * light_intensity.y * spec * shadow_factor,
        mat.specular.z * light_intensity.z * spec * shadow_factor
    )

    final_color = Point3(
        diffusion.x + specular.x + ambient.x,
        diffusion.y + specular.y + ambient.y,
        diffusion.z + specular.z + ambient.z
    )

    return Point3(
        min(1., max(0., final_color.x)),
        min(1., max(0., final_color.y)),
        min(1., max(0., final_color.z))
    )


def cross_vector_with_scene(scene_: list[SceneObject], ray: Ray, c_pos: Point3) -> [int, Point3]:
    closest_intersection: Point3 | None = None
    closest_object: int = -1
    min_distance = float("inf")

    for ind, so in enumerate(scene_):
        cross = so.obj.get_cross_point(ray)
        if cross is None:
            continue
        curr_distance = cross.distance(c_pos)
        if closest_intersection is None or curr_distance < min_distance:
            closest_intersection = cross
            closest_object = ind
            min_distance = curr_distance
    return closest_object, closest_intersection


def check_shadow(scene_: list[SceneObject], obj_num: int, pos: Point3, normal: Point3, ls: list[LightSource]) -> float:
    shadow_factor = 1.0

    for lgt_src in ls:
        light_dir = lgt_src.pos() - pos
        light_dist = light_dir.length()
        light_dir = light_dir.norm()

        ray_start = pos + normal * 0.001

        ray = Ray(ray_start, light_dir)

        in_shadow = False
        for i in range(len(scene_)):
            if i == obj_num:
                continue

            cross = scene_[i].obj.get_cross_point(ray)
            if cross is not None:

                cross_dist = (cross - ray_start).length()
                if cross_dist < light_dist:
                    in_shadow = True
                    break

        if in_shadow:
            shadow_factor = 0.0
            break

    return shadow_factor


class Sphere(Object):
    def __init__(self, radius: float, pos: Point3):
        self.r = radius
        self.pos = pos

    def get_cross_point(self, ray: Ray) -> Point3 | None:
        oc = Point3(ray.pos.x - self.pos.x, ray.pos.y - self.pos.y, ray.pos.z - self.pos.z)

        a = ray.rot.x ** 2 + ray.rot.y ** 2 + ray.rot.z ** 2
        b = 2 * (oc.x * ray.rot.x + oc.y * ray.rot.y + oc.z * ray.rot.z)
        c = oc.x ** 2 + oc.y ** 2 + oc.z ** 2 - self.r ** 2

        d = b ** 2 - 4 * a * c
        if d < 0:
            return None

        t1 = (-b - d ** 0.5) / (2 * a)
        t2 = (-b + d ** 0.5) / (2 * a)

        t = min(t1, t2) if t1 > 0 and t2 > 0 else max(t1, t2)
        if t < 0:
            return None

        return ray.get_point(t)

    def get_normal(self, p: Point3) -> Point3:
        return (p - self.pos).norm()

    def get_lightning_map(self, scene_: list[SceneObject], c_pos: Point3, ls: list[LightSource], mat: Material,
                          res: Point2, obj_ind: int) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, int(res.x), endpoint=False)
        phi = np.linspace(0, np.pi, int(res.y), endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x = self.r * np.sin(phi_grid) * np.cos(theta_grid)
        y = self.r * np.sin(phi_grid) * np.sin(theta_grid)
        z = self.r * np.cos(phi_grid)
        result = np.zeros((int(res.y), int(res.x), 3))
        for i in range(int(res.y)):
            for j in range(int(res.x)):
                pnt = Point3(x[i, j], y[i, j], z[i, j]) + self.pos
                normal = Point3(x[i, j], y[i, j], z[i, j]).norm()
                color = Point3(0, 0, 0)
                for light_src in ls:
                    material_color: Point3 = apply_blin_fong(
                        pnt,
                        normal,
                        c_pos,
                        light_src,
                        mat,
                        check_shadow(scene_, obj_ind, pnt, normal, ls),
                        self.map_to_xy(pnt)
                    )
                    color += material_color
                result[i, j] = [color.x * 255, color.y * 255, color.z * 255]
        return np.clip(result, 0, 255).astype(np.uint8)

    def map_to_xy(self, pnt: Point3) -> Point2:
        sphere_surf_pnt = pnt - self.pos
        r = sphere_surf_pnt.length()

        normalized = sphere_surf_pnt / r

        theta = math.acos(max(-1.0, min(1.0, normalized.z)))

        phi = math.atan2(normalized.y, normalized.x) + math.pi / 2

        u = phi / (2 * math.pi) + 0.5
        v = theta / math.pi

        return Point2(u, v)


class Camera:
    def __init__(self, pos: Point3, len: float, screen_size: Point2, screen_res: Point2):
        self.pos = pos
        self.screen_center = Point3(pos.x, pos.y + len, pos.z)
        self.s_size = screen_size
        self.s_res = screen_res

    def get_ray(self, pixel_pnt: Point2) -> Ray:
        x_d = ((pixel_pnt.x / self.s_res.x) - 0.5) * self.s_size.x
        y_temp = self.s_res.y - pixel_pnt.y
        y_d = ((y_temp / self.s_res.y) - 0.5) * self.s_size.y

        target = Point3(
            self.screen_center.x + x_d,
            self.screen_center.y,
            self.screen_center.z + y_d
        )

        direction = Point3(
            target.x - self.pos.x,
            target.y - self.pos.y,
            target.z - self.pos.z
        )
        direction.norm()

        return Ray(self.pos, direction)


def render(c_: Camera, scene_: list[SceneObject], ls: list[LightSource]) -> np.ndarray:
    ret = np.zeros((int(c_.s_res.y), int(c_.s_res.x), 3))
    for pix_y in range(int(c_.s_res.y)):
        for pix_x in range(int(c_.s_res.x)):
            pixel_point: Point2 = Point2(pix_x, pix_y)
            ray: Ray = c_.get_ray(pixel_point)

            closest_object_ind, closest_intersection = cross_vector_with_scene(scene_, ray, c_.pos)

            if (closest_object_ind == -1) or (closest_intersection is None):
                ret[pix_y, pix_x, 0] = 0
                ret[pix_y, pix_x, 1] = 0
                ret[pix_y, pix_x, 2] = 0
                continue

            closest_object = scene_[closest_object_ind]

            final_color = Point3(0, 0, 0)
            normal = closest_object.obj.get_normal(closest_intersection)
            for light_src in ls:
                material_color: Point3 = apply_blin_fong(
                    closest_intersection,
                    normal,
                    c_.pos,
                    light_src,
                    closest_object.mat,
                    check_shadow(scene_, closest_object_ind, closest_intersection, normal, ls),
                    closest_object.obj.map_to_xy(closest_intersection)
                )
                final_color += material_color

            final_color *= 255
            ret[pix_y, pix_x, 0] = final_color.x
            ret[pix_y, pix_x, 1] = final_color.y
            ret[pix_y, pix_x, 2] = final_color.z

    return ret.astype(np.uint8)


if __name__ == "__main__":
    c: Camera = Camera(Point3(0, 0, 0), 0.3, Point2(0.3, 0.3), Point2(100, 100))
    scene: list[SceneObject] = [
        SceneObject(
            Sphere(1., Point3(0, 5, 0)),
            Material(
                "img.png",
                Point3(1., 1., 1.),
                32
            )
        ),
        SceneObject(
            Sphere(.2, Point3(0, 3.5, 0)),
            Material(
                "img.png",
                Point3(1., 1., 1.),
                32
            )
        ),
        SceneObject(
            Sphere(.2, Point3(-1.5, 5, 0)),
            Material(
                "img.png",
                Point3(1., 1., 1.),
                32
            )
        ),
        SceneObject(
            Sphere(.2, Point3(0, 6.5, 0)),
            Material(
                "img.png",
                Point3(1., 1., 1.),
                32
            )
        ),
        SceneObject(
            Sphere(.2, Point3(1.5, 5, 0)),
            Material(
                "img.png",
                Point3(1., 1., 1.),
                32
            )
        ),
    ]
    ls: list[LightSource] = [
        LambertianLightSource(Point3(0, 1., 0), Point3(1., 1., 1.), Point3(-.3, 2 / 3, -2 / 3))
    ]
    fig, ax1 = plt.subplots()
    im1 = ax1.imshow(render(c, scene, ls))
    # im2 = ax2.imshow(scene[0].obj.get_lightning_map(scene, c.pos, ls, scene[0].mat, Point2(30, 30), 0))


    def update(frame):
        centered_light_pos = Point3(
            ls[0].position.x - scene[0].obj.pos.x,
            ls[0].position.y - scene[0].obj.pos.y,
            ls[0].position.z
        ).rotate(2, 0.1)
        ls[0].position = Point3(
            centered_light_pos.x + scene[0].obj.pos.x,
            centered_light_pos.y + scene[0].obj.pos.y,
            centered_light_pos.z)
        ls[0].normal = (scene[0].obj.pos - ls[0].position).norm()
        for so_idx in range(1, len(scene)):
            scene[so_idx].obj.pos = (scene[so_idx].obj.pos - scene[0].obj.pos).rotate(2, -.1) + scene[0].obj.pos
        im1.set_array(render(c, scene, ls))
        # im2.set_array(scene[0].obj.get_lightning_map(scene, c.pos, ls, scene[0].mat, Point2(30, 30), 0))
        return [im1]


    animation = FuncAnimation(fig, update, frames=100, interval=0, blit=True)
    plt.show()
