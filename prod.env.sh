case "${USER:-nouser}@${HOSTNAME:-nohost}" in

("thobson2@*")
    docker_base_container_name=th--${docker_base_container_name:?}
    docker_base_image_tag=th--${docker_base_image_tag:?}
    docker_prod_container_name=th--${docker_prod_container_name:?}
    docker_prod_image_tag=th--${docker_prod_image_tag:?}
    ;;

esac


scribe_root_dir=/opt
vainl_api_url=https://sdoh.with.vainl.in.production.is.mediocreatbest.xyz/vainl/

. "${root:?}/.prod.env"
